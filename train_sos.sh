be=([0]="1,4" [1]="2,6" [2]="3,9" [3]="5,15" [4]="8,16" [5]="13,26" [6]="21,42")

for gpu in 0 1 2 3 4 5 6; do
    iter=0
    pair=${be[$((gpu))]}
    b=${pair%,*}
    e=${pair#*,}
    epochs=8

    tmux new-session -d -s "train_gpu${gpu}" \
     "python -m scripts_sos.train_sos \
     --iter ${iter} --gpu ${gpu} --epochs ${epochs} --b ${b} --e ${e}"
  done