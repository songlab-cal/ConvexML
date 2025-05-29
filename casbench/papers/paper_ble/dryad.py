import os
from casbench.papers.paper_ble.slice_benchmark_paper_ble import slice_benchmark_paper_ble, get_ble_configs
from casbench import caching
from casbench.papers.paper_ble.global_variables import CACHE_PAPER_BLE
from casbench.simulated_data_benchmarking import run_missing_data_imputer_unrolled
from casbench import io
from casbench.config import smart_call
import tqdm

caching.set_cache_dir(CACHE_PAPER_BLE)
caching.set_dir_levels(3)
caching.set_log_level(9)
caching.set_read_only(True)

regimes = slice_benchmark_paper_ble(model_names=[], just_return_regimes_and_do_not_run=True)
print(regimes)
for (context_id, context_value, knob, knob_value, regime_args_list) in tqdm.tqdm(regimes):
    for repetition in range(50):
        output_tree_dir = smart_call(run_missing_data_imputer_unrolled, get_ble_configs(regime_args=dict(regime_args_list), model_name="gt__c__r__gt__gt", repetition=repetition))["output_tree_dir"]
        tree = io.read_tree(output_tree_dir + "/result.txt")
        tree_dir = f"trees/{knob}/{knob_value}/"
        os.makedirs(tree_dir, exist_ok=True)
        io.write_tree(tree, tree_dir + f"/tree_{repetition}_CassiopeiaTree.pkl")
        tree.character_matrix.to_csv(tree_dir + f"tree_{repetition}_character_matrix.csv")
        io.write_str(tree.get_newick(record_branch_lengths=True), tree_dir + f"tree_{repetition}_newick.txt")
        fitness_values = "\n".join([f"{leaf} {tree.get_attribute(leaf, 'fitness')}" for leaf in tree.leaves])
        io.write_str(fitness_values, tree_dir + f"tree_{repetition}_fitness.txt")
print("Done!")
