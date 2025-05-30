sleep_seconds=30

be=([0]="1,4" [1]="2,6" [2]="3,9" [3]="5,15" [4]="8,16" [5]="13,26" [6]="21,42")
max_tokens=([0]="200" [1]="300" [2]="500" [3]="1000" [4]="2000" [5]="3000" [6]="5000")

for port in 0 1 2; do
  for gpu in 0 1 2 3 4 5 6; do
    iter=1
    pair=${be[$((gpu))]}
    b=${pair%,*}
    e=${pair#*,}
    epochs=$(( 16 + (8 * port) ))
    tokens=${max_tokens[$gpu]}

    tmux new-session -d -s "sos_gpu${gpu}_port${port}" \
      "python -m scripts_sos.deploy_sos \
      --iter ${iter} --gpu ${gpu} --port ${port} --b ${b} --e ${e} --epochs ${epochs} --tokens ${tokens}"
  done
  sleep ${sleep_seconds}
done