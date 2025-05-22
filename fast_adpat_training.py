import json
import random
import textwrap
from fast_adapt_dataloader import get_dataset_dict, PretrainedTokenizerProcessor, dict_to_tuple_collate, custom_collate_fn

from fast_adapt_utils import compute_client_weights, test_inference, Client, filter_dataset_by_classes, partition_and_schedule, zero_state_dict_cpu, get_learning_rate_from_config, require_init_lr, require_StepLR_stepsize, require_StepLR_gamma, create_model, calculate_polynomial_decay_lr_schedule, get_init_lr_step_gamma_from_config, calculate_step_lr_schedule, get_lr_schedule_from_config, require_lr_min_ratio, initial_model_construction, run_pilot_stage, maybe_download

from fast_adpat_parser import parse_arguments
import time
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import copy
import matplotlib.ticker as mtick
from datetime import datetime
import pytz
from torch.utils.data import DataLoader, Subset
from pprint import pprint
from datasets import load_dataset, Features, ClassLabel, Value



def main_training_loop(args, client_list, type_testing_dataset_dict, training_order_epoch_task_list, run_identifier, skip_pilot, skip_recovery,  skip_construction, skip_average, continuous_baseline):
    
    dataset_name = args.dataset_name

    args.init_lr, args.lr_min_ratio = get_lr_schedule_from_config(args)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not found! Please ensure that a GPU is available.")
    gpu_device = torch.device(f"cuda:{args.gpu_index}")

    num_sessions_pilot = args.num_sessions_pilot

    num_sessions = args.num_sessions

    num_rounds_pilot = args.num_rounds_pilot

    num_rounds_actual = args.num_rounds_actual

    if num_rounds_pilot < 1:
        raise ValueError("num_rounds_pilot should be at least 1")
    
    if num_rounds_actual < 1:
        raise ValueError("num_of_iterations_client_fixed should be at least 1")

    global_model = create_model(args)

    initial_random_global_model = global_model.state_dict()

    alg_global_var_dict = {args.algorithm: zero_state_dict_cpu(global_model)}

    similarity_normalized_value_check_dict = {}
    similarity_original_value_check_dict = {}

    params = dict(global_model.named_parameters())
    num_computed_grads = num_sessions - num_sessions_pilot

    computed_grads = {
        name: torch.zeros(
            (num_computed_grads, *param.shape),
            dtype=torch.float32,
            device="cpu"  
        )
        for name, param in params.items()
    }

    init_warm_global_model_weight = {
        k: torch.zeros(
            (num_sessions, *v.shape),
            dtype=v.dtype,
            device=v.device
        )
        for k, v in global_model.state_dict().items()
    }

    accuracy_list = np.zeros(len(training_order_epoch_task_list)+1, dtype=float) 

    collate_fn_mapping = {"ag_news": dict_to_tuple_collate}

    collate_fn = collate_fn_mapping.get(args.dataset_name, None)

    initial_test_dataloader = DataLoader(type_testing_dataset_dict[args.dataset_name], batch_size= args.batch_size_training, shuffle=False, collate_fn= collate_fn)

    init_global_acc, _ = test_inference(args, gpu_device, global_model, initial_test_dataloader)

    accuracy_list[0] = init_global_acc

    average_warm_model_as_pilot_model_dict = {
    name: tensor.detach()
                  .to(dtype=torch.float32,   # explicitly float
                      device='cpu')          # explicitly on CPU
    for name, tensor in initial_random_global_model.items()
    }

    pilot_stage = training_order_epoch_task_list[:num_rounds_pilot * num_sessions_pilot]

    application_stage = training_order_epoch_task_list[num_rounds_pilot * num_sessions_pilot:]

    if not skip_pilot:
        run_pilot_stage(args, client_list, pilot_stage, global_model, init_warm_global_model_weight, accuracy_list, type_testing_dataset_dict)

    for sync_idx, current_client_idx_list in enumerate(application_stage):

        sync_idx = sync_idx + num_rounds_pilot * num_sessions_pilot
        
        client_weight_training = compute_client_weights(client_list, current_client_idx_list, "training_loader")
        client_weight_grad_cal = compute_client_weights(client_list, current_client_idx_list, "grad_cal_loader")

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            num_changes_in_set_of_device = sync_idx//num_rounds_pilot
        else:
            num_changes_in_set_of_device = (sync_idx - num_rounds_pilot * num_sessions_pilot)//num_rounds_actual + num_sessions_pilot

        change_point_indicator = False

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            if sync_idx%num_rounds_pilot == 0:
                change_point_indicator = True
        else:
            if (sync_idx - num_rounds_pilot * num_sessions_pilot)%num_rounds_actual == 0:
                change_point_indicator = True

        if change_point_indicator:

            lr_list = calculate_polynomial_decay_lr_schedule(
                initial_client_lr = require_init_lr(args),
                total_communication_rounds = args.num_rounds_actual,
                power = 0.9,
                min_client_lr = require_init_lr(args) * require_lr_min_ratio(args)
            )

            alg_global_var_dict = {args.algorithm: zero_state_dict_cpu(global_model)}
            
            if not skip_recovery:
                global_model.load_state_dict(initial_random_global_model)

        current_lr = lr_list[sync_idx%num_rounds_actual]

        print("")
        print("sync_idx", sync_idx, "current_lr", current_lr)

        if not skip_construction:

            initial_model_construction(args, client_list, sync_idx, num_changes_in_set_of_device, change_point_indicator, init_warm_global_model_weight, average_warm_model_as_pilot_model_dict, current_client_idx_list, client_weight_grad_cal, computed_grads, global_model, similarity_normalized_value_check_dict, similarity_original_value_check_dict)
        
        if not skip_average:

            if num_changes_in_set_of_device >= (1 + num_sessions_pilot) and change_point_indicator:
                
                # 1) Determine the range of “warm” models to average
                start = num_sessions_pilot
                end   = num_changes_in_set_of_device
                count = end - start
                inv_count = 1.0 / count

                # 2) Pre-allocate CPU buffers for the averaged global (and scaffold C, if needed)
                avg_global = {
                    k: torch.zeros_like(v.detach().cpu(), dtype=torch.float32)
                    for k, v in global_model.state_dict().items()
                }

                # 3) Fold each historical model into the average (in-place, fused mul+add)
                for px in range(start, end):
                    
                    for k, batched in init_warm_global_model_weight.items():
                        # `batched` has shape (num_sessions, *param.shape)
                        # batched[px] is exactly the old global at session px
                        avg_global[k].add_(batched[px], alpha=inv_count)

                # 4) Overwrite model (and scaffold‐C) in one go
                global_model.load_state_dict(avg_global)

        if sync_idx == num_rounds_pilot * num_sessions_pilot:
            global_model.load_state_dict(initial_random_global_model) 

        unique_classes = set()

        # 1) pull your global parameters down to CPU once
        cpu_state_dict = {
            k: v.detach().cpu()
            for k, v in global_model.state_dict().items()
        }

        # 2) init an accumulator of the same shape on CPU

        new_global = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in cpu_state_dict.items()
        }

        if continuous_baseline:
            
            for client_idx in current_client_idx_list:

                if change_point_indicator:

                    if num_changes_in_set_of_device == 0:

                        client_list[client_idx].set_model_last_session(initial_random_global_model)

                    else:

                        prev_idx = num_changes_in_set_of_device - 1
                        if prev_idx < 0:
                            raise ValueError(f"No previous session (s={num_changes_in_set_of_device})")

                        # 1) Build a state_dict for session s–1, ensuring all tensors are on CPU
                        prev_state_dict = {
                            k: buf[prev_idx].cpu()
                            for k, buf in init_warm_global_model_weight.items()
                        }

                        client_list[client_idx].set_model_last_session(prev_state_dict)
                    
                print(f'Client Index {client_idx}')
                local_model_parameter_dict = client_list[client_idx].local_training_continuous(
                    global_model.state_dict(), current_lr
                    )
                
                w = client_weight_training[client_idx]

                for k, v_local in local_model_parameter_dict.items():
                    # new_global[k] += w * v_local  (all on CPU)
                    new_global[k].add_(v_local, alpha=w)

                unique_classes.update(client_list[client_idx].class_set)
        
        else:

            if args.algorithm == "avg":

                for client_idx in current_client_idx_list:
                    
                    print(f'Client Index {client_idx}')
                    local_model_parameter_dict = client_list[client_idx].local_training_avg(
                        global_model.state_dict(), current_lr
                        )
                    
                    w = client_weight_training[client_idx]

                    for k, v_local in local_model_parameter_dict.items():
                        # new_global[k] += w * v_local  (all on CPU)
                        new_global[k].add_(v_local, alpha=w)

                    unique_classes.update(client_list[client_idx].class_set)
            
            elif args.algorithm == "prox":

                for client_idx in current_client_idx_list:
                
                    print(f'Client Index {client_idx}')
                    local_model_parameter_dict = client_list[client_idx].local_training_prox(
                        global_model.state_dict(), current_lr
                        )
                    
                    w = client_weight_training[client_idx]

                    for k, v_local in local_model_parameter_dict.items():
                        # new_global[k] += w * v_local  (all on CPU)
                        new_global[k].add_(v_local, alpha=w)

                    unique_classes.update(client_list[client_idx].class_set)
            
            elif args.algorithm == "scaffold":

                accum_y = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in cpu_state_dict.items() }
                accum_c = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in cpu_state_dict.items() }

                # loop once over all clients, doing both accumulations
                for client_idx in current_client_idx_list:
                    print(f'Client Index {client_idx}')
                    local_dy, local_dc = client_list[client_idx].local_training_scaffold(
                        global_model.state_dict(), alg_global_var_dict[args.algorithm], current_lr
                    )
                    w = client_weight_training[client_idx]

                    # in-place add: accum += w * local_delta
                    for k, delta_y in local_dy.items():
                        accum_y[k].add_(delta_y, alpha=w)
                    for k, delta_c in local_dc.items():
                        accum_c[k].add_(delta_c, alpha=w)

                    unique_classes.update(client_list[client_idx].class_set)

                new_global = {
                    k: cpu_state_dict[k] + current_lr * accum_y[k]
                    for k in cpu_state_dict
                }

                for k, delta in accum_c.items():
                    alg_global_var_dict[args.algorithm][k].add_(delta)
                
            elif args.algorithm == "moon":

                for client_idx in current_client_idx_list:
                    
                    print(f'Client Index {client_idx}')
                    local_model_parameter_dict = client_list[client_idx].local_training_moon(
                        global_model.state_dict(), current_lr
                        )
                    
                    w = client_weight_training[client_idx]

                    for k, v_local in local_model_parameter_dict.items():
                        # new_global[k] += w * v_local  (all on CPU)
                        new_global[k].add_(v_local, alpha=w)

                    unique_classes.update(client_list[client_idx].class_set)
            
            elif args.algorithm == "acg":

                delta_bar = {k: torch.zeros_like(v, dtype=torch.float32) for k,v in cpu_state_dict.items()}

                for client_idx in current_client_idx_list:
                    delta_c = client_list[client_idx].local_training_acg(
                        global_model.state_dict(), alg_global_var_dict[args.algorithm], current_lr
                    )               
                    w = client_weight_training[client_idx]

                    unique_classes.update(client_list[client_idx].class_set)

                    for k, p_c in delta_c.items():
                        delta_bar[k].add_(p_c, alpha=w)
    
                for k in alg_global_var_dict[args.algorithm]:
                    alg_global_var_dict[args.algorithm][k].mul_(getattr(args, 'acg_lambda', 0.85))
                    alg_global_var_dict[args.algorithm][k].add_(delta_bar[k])

                # 3) form new global state on CPU
                new_global = {
                    k: v.cpu().clone() + alg_global_var_dict[args.algorithm][k] for k, v in global_model.state_dict().items()
                }

        save_warm_model_indicator = False

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            if sync_idx%num_rounds_pilot == num_rounds_pilot - 1:
                save_warm_model_indicator = True
        else:
            if (sync_idx - num_rounds_pilot * num_sessions_pilot)%num_rounds_actual == num_rounds_actual - 1:
                save_warm_model_indicator = True
        
        if save_warm_model_indicator:

            session_idx = num_changes_in_set_of_device

            if session_idx < num_sessions:
                # 1) Store the new global weights into the batched buffer
                #    `init_warm_global_model_weight[k]` is a Tensor of shape (num_sessions, *shape)
                for k, buf in init_warm_global_model_weight.items():
                    # `new_global[k]` should be a CPU tensor matching buf’s [session_idx].shape
                    buf[session_idx].copy_(new_global[k].cpu())

        global_model.load_state_dict(new_global) 

 
        test_dataloader = filter_dataset_by_classes(type_testing_dataset_dict[dataset_name], unique_classes, args.batch_size_training, dataset_name)

        acc, global_loss = test_inference(args, gpu_device, global_model, test_dataloader)

        accuracy_list[sync_idx+1] = acc

    if not skip_construction:

        similarity_values_dir_path = os.path.join(os.getcwd(), args.similarity + "_value", args.dataset_name)
        
        if not os.path.exists(similarity_values_dir_path):
            os.makedirs(similarity_values_dir_path)

        similarity_values_dt_dir_path = os.path.join(similarity_values_dir_path, f"{run_identifier}")
        if not os.path.exists(similarity_values_dt_dir_path):
            os.makedirs(similarity_values_dt_dir_path)

        with open(os.path.join(similarity_values_dt_dir_path,f'original.json'), 'w') as f:
            json.dump(similarity_original_value_check_dict,f, indent= 4)

        with open(os.path.join(similarity_values_dt_dir_path,f'normalized.json'), 'w') as f:
            json.dump(similarity_normalized_value_check_dict,f, indent= 4)

    return accuracy_list


if __name__ == "__main__":
    

    args = parse_arguments()

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    job_id = os.environ.get('SLURM_JOB_ID')

    if job_id:
        # Running under SLURM, use the job ID (Best practice)
        run_identifier = f"job_{job_id}"
        print(f"Running as SLURM Job ID: {job_id}")
    else:
        eastern_tz = pytz.timezone('America/New_York')
        utc_now = datetime.datetime.utcnow()
        utc_aware_now = pytz.utc.localize(utc_now) # Make the UTC time timezone-aware
        eastern_now = utc_aware_now.astimezone(eastern_tz) # Convert to Eastern Time
        process_id = os.getpid()
        run_identifier = f"local_ET_{eastern_now}_pid_{process_id}"

    program_start_time = time.time()

    num_sessions = args.num_sessions
    num_clients = args.num_clients
    num_sessions_pilot = args.num_sessions_pilot

    num_rounds_pilot = args.num_rounds_pilot
    num_rounds_actual = args.num_rounds_actual

    if num_sessions_pilot > num_sessions:
        raise ValueError("num_sessions_pilot should no larger than num_sessions!")

    if args.dataset_name == "ag_news":

        shared_dir = os.path.join(os.getcwd(), "data", "ag_news")
        train_csv = os.path.join(shared_dir, "train.csv")
        test_csv  = os.path.join(shared_dir, "test.csv")

        # 2) URLs for AG News
        URL_TRAIN = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        URL_TEST  = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        # 3) Ensure they're downloaded once
        maybe_download(URL_TRAIN, train_csv)
        maybe_download(URL_TEST,  test_csv)

        # Load AG News
        dataset = load_dataset(
            "csv",
            data_files={"train": train_csv, "test": test_csv},
            column_names=["label", "title", "description"],  # CSV has columns in this order
        )

        # 2) define the exact HF features you want:
        features = Features({
            "text":  Value("string"),
            "label": ClassLabel(names=["World","Sports","Business","Sci/Tech"])
        })

        # 3) map→ concat title+desc, shift labels from 1–4 → 0–3, remove raw cols, and apply your features
        def _prep(example):
            return {
                "text":  example["title"] + " " + example["description"],
                "label": example["label"] - 1,
            }

        dataset = dataset.map(
            _prep,
            remove_columns=["title","description"],
            features=features
        )

        # Preprocess using the processor
        processor = PretrainedTokenizerProcessor(model_name="bert-base-uncased", max_len=64)
        processed_train = processor.apply(dataset["train"])
        processed_test = processor.apply(dataset["test"])

        # Store the PROCESSED dataset directly (NO WRAPPER)
        type_training_dataset_dict = {args.dataset_name: processed_train}
        type_testing_dataset_dict = {args.dataset_name: processed_test}

        args.vocab_size = processor.vocab_size
        args.pad_idx = processor.pad_idx

    else:
        type_training_dataset_dict = get_dataset_dict(args, train=True)
        type_testing_dataset_dict = get_dataset_dict(args, train=False)

    # if args.dataset_name == "ag_news":

    #     if args.label_distribution == "half":
    #         clients_data_ids, client_classes = distinct_half_agnews(type_training_dataset_dict[args.dataset_name], args.num_clients)

    #     elif args.label_distribution == "partial_overlap":

    #         clients_data_ids, client_classes = distribute_agnews_labels_specific(type_training_dataset_dict[args.dataset_name], total_clients= args.num_clients)
        
    #     elif args.label_distribution == "distinct":
    #         clients_data_ids, client_classes = distribute_agnews_exclusive_quarters(type_training_dataset_dict[args.dataset_name], args.num_clients)

    #     elif args.label_distribution == "two_shard":
    #         clients_data_ids, client_classes = pathological_non_iid_partition(type_training_dataset_dict[args.dataset_name], args.num_clients)


    # elif args.label_distribution == "half":
    #     clients_data_ids, client_classes = distinct_half(type_training_dataset_dict[args.dataset_name], args.num_clients)

    # elif args.label_distribution == "distinct":
    #     clients_data_ids, client_classes = distinct_class_each_device(type_training_dataset_dict[args.dataset_name])
    
    # elif args.label_distribution == "two_shard":

    #     clients_data_ids, client_classes = distribute_labels_in_batches(type_training_dataset_dict[args.dataset_name], args.num_clients)
    
    # elif args.label_distribution == "partial_overlap":

    #     clients_data_ids, client_classes = distribute_labels_slight_overlap_clients(type_training_dataset_dict[args.dataset_name], total_clients= args.num_clients)

    # else:
    #     raise ValueError("Wrong values to class distribution!")

    clients_data_ids, client_classes, session_clients, training_order = partition_and_schedule(
    dataset = type_training_dataset_dict[args.dataset_name],
    num_clients_per_group = num_clients,
    num_sessions = num_sessions,
    num_sessions_pilot = num_sessions_pilot,
    num_rounds_pilot = num_rounds_pilot,
    num_rounds_actual = num_rounds_actual,
    cross_session_label_overlap = args.cross_session_label_overlap,
    in_session_label_dist = args.in_session_label_dist,
    dirichlet_alpha = args.dirichlet_alpha,
    shards_per_client = 2,
    unbalanced_sgm = 0.0,
    )
    
    # 1) grab the raw dataset once
    ds = type_training_dataset_dict[args.dataset_name]

    # 2) decide if we need the special collate_fn
    collate_fn_mapping = {"ag_news": dict_to_tuple_collate}

    collate_fn = collate_fn_mapping.get(args.dataset_name, None)

    def make_loader(indices, batch_size):
        kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
        }
        if collate_fn:
            kwargs["collate_fn"] = collate_fn
        return DataLoader(Subset(ds, indices), **kwargs)

    # 3a) via list comprehension
    client_list = [
        Client(
            args,
            make_loader(clients_data_ids[i], args.batch_size_training),
            make_loader(clients_data_ids[i], args.batch_size_grad_cal),
            client_classes[i],
        )
        for i in range(num_clients * 2)
    ]

    results_dict = {}

    pprint(vars(args))

    if args.initial:

        print("Running initial model construction...") 
        results_dict["initial"] = main_training_loop(args, client_list, type_testing_dataset_dict, training_order, run_identifier, skip_pilot = False, skip_recovery = True,  skip_construction = False, skip_average = True, continuous_baseline = False)
        print("Initial model construction finished.")

    if args.recovery:
        print("Running recovery baseline...") 
        results_dict["recovery"] = main_training_loop(args, client_list, type_testing_dataset_dict, training_order, run_identifier, skip_pilot = True, skip_recovery = False,  skip_construction = True, skip_average = True, continuous_baseline = False)
        print("Recovery baseline finished.")

    if args.average:
        print("Running average baseline...") 
        results_dict["average"] = main_training_loop(args, client_list, type_testing_dataset_dict, training_order, run_identifier, skip_pilot = True, skip_recovery = True,  skip_construction = True, skip_average = False, continuous_baseline = False)
        print("Average baseline finished.")

    if args.continuous:
        print("Running continuous baseline...") 
        results_dict["continuous"] = main_training_loop(args, client_list, type_testing_dataset_dict, training_order, run_identifier, skip_pilot = True, skip_recovery = True,  skip_construction = True, skip_average = True, continuous_baseline = True)
        print("Continuous baseline finished.")

    if args.previous:
        print("Running previous baseline...") 
        results_dict["previous"] = main_training_loop(args, client_list, type_testing_dataset_dict, training_order, run_identifier, skip_pilot = True, skip_recovery = True,  skip_construction = True, skip_average = True, continuous_baseline = False)
        print("Previous baseline finished.")

    # Define plot styles and labels for each result type
    plot_info = {
        "initial": {"label": "With Initial Point Selection", "marker": "p", "linestyle": "--", "linewidth": 3},
        "recovery": {"label": "Recovery", "marker": "o", "linestyle": "--", "linewidth": 3},
        "average": {"label": "Avg Baseline", "marker": "o", "linestyle": "--", "linewidth": 3},
        "continuous": {"label": "Continuous", "marker": "o", "linestyle": "--", "linewidth": 3},
        "previous": {"label": "Previous Baseline", "marker": "o", "linestyle": "--", "linewidth": 3}, # Added style for previous
    }

    # Define common xticks calculation
    xticks_np_range = np.array([num_rounds_pilot] * num_sessions_pilot +
                            [num_rounds_actual] * (num_sessions - num_sessions_pilot))
    xticks_positions = np.cumsum(xticks_np_range)

    # Loop through results_dict and plot/save each result type
    print("Starting plotting and saving...") # Added print for clarity
    for key, data in results_dict.items():
        # Ensure key exists in plot_info before proceeding, although with the current logic it should
        if key in plot_info and data is not None and len(data) > 0:
            print(f"Processing results for: {key}")

            if args.in_session_label_dist == "dirichlet":
                dist_info = rf"dirichlet($\alpha$={args.dirichlet_alpha})"
            elif args.in_session_label_dist == "two_shards":
                dist_info = f"two_shards"
            else:
                dist_info = args.in_session_label_dist

            raw_title = (
                f"{args.dataset_name}, {args.algorithm}, "
                f"label_overlap={args.cross_session_label_overlap}, "
                f"in_session_label_dist={dist_info}, "
                f"seed={args.seed}, "
            )

            # --- Plotting ---
            plt.figure()
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            style = plot_info[key] # Get plot style for the current key

            plt.step(range(len(data)), data,
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=3,
                    label=style["label"])

            plt.xticks(xticks_positions)
            plt.xticks(fontsize=8)
            ax.legend()
            wrapped_title = "\n".join(textwrap.wrap(raw_title, width=60))
            ax.set_title(wrapped_title, fontsize=12, pad=6)

            fig = plt.gcf()
            fig.tight_layout()            # use full figure
            fig.subplots_adjust(top=0.90) # shrink top margin to 10%

            # Define and create plot directory
            training_plot_dir_path = os.path.join(os.getcwd(), "training_plot", args.dataset_name)
            if not os.path.exists(training_plot_dir_path):
                os.makedirs(training_plot_dir_path)

            # Save the plot with key in filename to avoid overwriting
            plt.savefig(os.path.join(training_plot_dir_path, f"{run_identifier}_{key}.png"))
            plt.close() # Close the figure to free up memory

            # --- Saving Training Values ---
            # Define and create training values directory
            training_value_time_dir_path = os.path.join(os.getcwd(), "training_values", args.dataset_name, f"{run_identifier}")
            if not os.path.exists(training_value_time_dir_path):
                os.makedirs(training_value_time_dir_path)

            # Save the result data as JSON
            with open(os.path.join(training_value_time_dir_path, f'{key}.json'), 'w') as f:
                json.dump(data.tolist(), f, indent=4)

    print("Plotting and saving finished for all results.")

    # 3. Calculate and save elapsed time (done only once at the end)
    end_time = time.time()
    elapsed_time = end_time - start_time

    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    elapsed_time_string = f"{days}d {hours}h {minutes}m {seconds:.2f}s"
    args.elapsed_time = elapsed_time_string # Add elapsed time to args

    print(f"Total elapsed time: {elapsed_time_string}")

    # 4. Save args and training order (done only once at the end)

    # Save args namespace
    argument_dir_path = os.path.join(os.getcwd(), "args_namespace", args.dataset_name)
    if not os.path.exists(argument_dir_path):
        os.makedirs(argument_dir_path)
    with open(os.path.join(argument_dir_path, f'{run_identifier}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Save training order
    training_order_dir_path = os.path.join(os.getcwd(), "training_order")
    if not os.path.exists(training_order_dir_path):
        os.makedirs(training_order_dir_path)
    with open(os.path.join(training_order_dir_path, f"{run_identifier}.json"), 'w') as f:
        json.dump(session_clients, f, indent=4)

    print("Args and training order saved.")


