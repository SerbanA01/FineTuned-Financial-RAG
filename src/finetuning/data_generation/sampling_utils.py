# src/finetuning/data_generation/sampling_utils.py

import random

def get_random_non_overlapping_groups(data: list, group_size: int, k: int) -> list:
    """
    Extracts k random, non-overlapping, consecutive groups of a specified size from a list.

    This function is useful for creating batches of related context for tasks like
    synthetic data generation, where each batch (group) should be distinct from
    the others. It ensures that no single item from the input data appears in more
    than one output group.

    For example, given `data = [0, 1, 2, 3, 4, 5, 6, 7]`, `group_size = 3`, and `k = 2`,
    a possible valid output would be `[[1, 2, 3], [5, 6, 7]]`. The group `[2, 3, 4]`
    would be invalid to select after `[1, 2, 3]` because it overlaps.

    @param data: The list of items to sample from (e.g., lines from a file, text chunks).
    @param group_size: The number of consecutive items to include in each group.
    @param k: The maximum number of non-overlapping groups to extract.

    @return: A list containing up to k groups. Each group is a list of
             `group_size` consecutive items from the original data. If not enough
             non-overlapping groups can be formed, the list may contain fewer than k groups.
    """
    # 1. Determine all possible starting positions for a group of `group_size`.
    # If the data is smaller than the group size, no groups can be formed.
    if len(data) < group_size:
        return []
    # A group can start at any index from 0 up to the point where there are
    # exactly `group_size` elements remaining.
    possible_starts = list(range(len(data) - group_size + 1))
    
    # 2. Shuffle the potential starting indices. This randomization is key to ensuring
    # that the selected groups are not biased towards the beginning of the data.
    random.shuffle(possible_starts)
    
    # This set will keep track of which indices from the original `data` list
    # have already been assigned to a group.
    used_indices = set()
    result_groups = []
    
    # 3. Iterate through the randomized starting points to build the result set.
    for start_index in possible_starts:
        # Stop once the desired number of groups (`k`) has been collected.
        if len(result_groups) >= k:
            break
            
        # Define the full range of indices that this potential group would occupy.
        group_indices = range(start_index, start_index + group_size)
        
        # 4. Check for any overlap with indices that are already in `used_indices`.
        # The `any()` function provides an efficient way to check for intersection.
        is_overlapping = any(i in used_indices for i in group_indices)
        
        # If there's no overlap, the group is valid.
        if not is_overlapping:
            # Extract the group from the original data using slicing.
            group = data[start_index : start_index + group_size]
            result_groups.append(group)
            
            # Mark the indices of this new group as used to prevent them from
            # being included in any future groups.
            used_indices.update(group_indices)
            
    return result_groups