for dataset in wisconsin 
do
    for hidden_channels in 1024 512 256
    do
        for num_centers in  8 10 20 30 40
        do
            for lr in 0.0001 0.0005
            do
                for dropout in 0 0.1 0.2 0.35
                do
                    # We show in terminal the message: "Running $dataset with $hidden_channels hidden channels and $num_centers centers"
                    echo "Running $dataset with $hidden_channels hidden channels and $num_centers centers and $lr learning rate and $dropout dropout"
                    python main.py --dataset $dataset --hidden_channels $hidden_channels --num_centers $num_centers --epochs 200 --lr $lr  --dropout $dropout 
                    # Now we print the last row of the csv results.csv
                    echo $(tail -n 1 results.csv)
                done
            done
        done
    done
done