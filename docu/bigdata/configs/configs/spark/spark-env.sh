#!/usr/bin/env bash
##
# Generated by Cloudera Manager and should not be modified directly
##
export SPARK_HOME=/home/hadoop/spark

export HADOOP_HOME=/home/hadoop/core

export HADOOP_COMMON_HOME="$HADOOP_HOME"

export SPARK_WORKER_MEMORY=1024MB

#export spark.yarn.archive=hdfs://hadoop3:2220/user/spark/jars/spark-libs.jar
export SPARK_DRIVER_MEMORY=5G

export SPARK_BEELINE_MEMORY=1024MB
export SPARK_EXECUTOR_MEMORY=1024MB
export SPARK_MASTER_HOST=hadoop2
export SPARK_MASTER_PORT=7077

PYLIB="$SPARK_HOME/python/lib"
if [ -f "$PYLIB/pyspark.zip" ]; then
  PYSPARK_ARCHIVES_PATH=
  for lib in "$PYLIB"/*.zip; do
    if [ -n "$PYSPARK_ARCHIVES_PATH" ]; then
      PYSPARK_ARCHIVES_PATH="$PYSPARK_ARCHIVES_PATH,local:$lib"
    else
      PYSPARK_ARCHIVES_PATH="local:$lib"
    fi
  done
  export PYSPARK_ARCHIVES_PATH
fi
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop

export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
export JAVA_LIBRARY_PATH=$HADOOP_HOME/lib//native

export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://hadoop3:2220/sparklog/ -Dspark.history.fs.cleaner.enabled=true"