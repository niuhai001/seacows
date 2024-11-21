
# Hadoop: HDFS & YARN
for m in 0 1 2 3 4; do
   echo "dist hadoop conf to hadoop$m"
   rsync -avz hadoop/ root@hadoop$m:/home/hadoop/core/etc/hadoop
done

# Hive
for m in 0 1 2 3 4 ; do 
   echo "dist hive conf to hadoop$m"
   rsync -avz hive/ root@hadoop$m:/home/hadoop/hive/conf
done

# Spark
for m in 0 1 2 3 4; do 
   echo "dist spark conf to hadoop$m"
   rsync -avz spark/ root@hadoop$m:/home/hadoop/spark/conf
done
#kafka
#for m in  2 3 4; do 
#   echo "dist kafka conf to hadoop$m"
#   rsync -avz kafka/ root@hadoop$m:/home/hadoop/kafka/config
#done
