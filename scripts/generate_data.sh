# Examples:
# bash scripts/generate_data.sh "demos/test" 10 2

dataset_path=${1}
category_one_num_replays=${2}
category_two_num_replays=${3}

echo -e "\033[36mStarting data generation... (҂◡_◡) ᕤ\033[0m"

# Replay trajectories for first category
for file in ${dataset_path}/category_a/*.h5; do
    [ -e "$file" ] || continue
    echo -e "\033[35mReplaying "$file" for the first category...\033[0m"
    python -m ManiSkill.data_collection.replay_trajectory \
                            --traj-path "$file" \
                            --count ${category_one_num_replays} \
                            --save-traj \
                            -o "pointcloud"
    python generate_data.py --dataset-path "$file" \
                            --directory ${dataset_path} \
                            --replay-num ${category_one_num_replays}
done
echo -e "\033[32mSuccess!\033[0m"

# Replay trajectories for second category
for file in ${dataset_path}/category_b/*.h5; do
    [ -e "$file" ] || continue
    echo -e "\033[35mReplaying "$file" for the second category...\033[0m"
    python -m ManiSkill.data_collection.replay_trajectory \
                            --traj-path "$file" \
                            --count ${category_two_num_replays} \
                            --save-traj \
                            -o "pointcloud"
    python generate_data.py --dataset-path "$file" \
                            --directory ${dataset_path} \
                            --replay-num ${category_two_num_replays}
done
echo -e "\033[32mSuccess!\033[0m"
