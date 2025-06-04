# gen_sos_data.sh

be=([0]="1,4" [1]="2,6" [2]="3,9" [3]="5,15" [4]="8,16" [5]="13,26" [6]="21,42")

for port in 0 1; do
  for gpu in 6; do
    iter=7
    pair=${be[$((gpu))]}
    b=${pair%,*}
    e=${pair#*,}

    tmux new-session -d -s "gen_gpu${gpu}_port${port}" \
     "python -m scripts_mcts.generate_data \
     --iter ${iter} --gpu ${gpu} --port ${port} --b ${b} --e ${e} ; bash"
  done
done
