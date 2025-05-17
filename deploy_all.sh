#deploy_all.sh

sleep_seconds=30

for port in 0 1; do
  for gpu in {1..7}; do
    policy_iter=$gpu
    # policy deploy
    tmux new-session -d -s "policy_iter${policy_iter}_port${port}" \
      "python -m scripts.deploy_policy --iter ${policy_iter} --gpu ${gpu} --port ${port}"
  done
  sleep ${sleep_seconds}
  for gpu in {1..7}; do
    value_iter=$gpu
    # value deploy
    tmux new-session -d -s "value_iter${value_iter}_port${port}" \
      "python -m scripts.deploy_value  --iter ${value_iter} --gpu ${gpu} --port ${port}"
  done
  sleep ${sleep_seconds}
done