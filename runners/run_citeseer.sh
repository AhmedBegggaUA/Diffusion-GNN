for dataset in citeseer 
do
    for hidden_channels in 64 128 256
    do
        for num_centers in 3 4 6 5 8 10 # 12 20 30
        do
            for lr in 0.01 0.001
            do
                for dropout in 0 0.1 0.2 0.35
                do
                    # We show in terminal the message: "Running $dataset with $hidden_channels hidden channels and $num_centers centers"
                    echo "Running $dataset with $hidden_channels hidden channels and $num_centers centers and $lr learning rate and $dropout dropout"
                    python main.py --dataset $dataset --hidden_channels $hidden_channels --num_centers $num_centers --epochs 500 --lr $lr  --dropout $dropout 
                    # Now we print the last row of the csv results.csv
                    echo $(tail -n 1 results.csv)
                done
            done
        done
    done
done