#! /bin/bash -l

if [[ "$#" -ne 1 ]]; then
    echo "Illegal number of parameters. Need to provide either 'start' or 'kill'"
fi


if [[ "$1" = "--start" ]];
then

    # Initialize Federation Evaluator
    echo "Initializing Evaluator Service at BDNF"
    EVALUATOR_SERVICE_CUDA_DEVICE="5"
    EVALUATOR_SERVICE_PS_PORT=8989
    EVALUATOR_SERVICE_WORKER_PORT=8990
    ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"$EVALUATOR_SERVICE_CUDA_DEVICE\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
    nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name ps --task_index 0 --port_num ${EVALUATOR_SERVICE_PS_PORT} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/ps${EVALUATOR_SERVICE_PS_PORT}.out 2>&1 &"
    ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"$EVALUATOR_SERVICE_CUDA_DEVICE\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
    nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name worker --task_index 0 --port_num ${EVALUATOR_SERVICE_WORKER_PORT} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/worker${EVALUATOR_SERVICE_WORKER_PORT}.out 2>&1 &"

    # ps_ports for 10 Fast on bdnf: "2222 2224 2226 2228 2230 2232 2234 2236 2238 2240"
    # ps_ports for 5 Fast, 5 Slow on bdnf and learn: "2222 2224 2226 2228 2230"
    # ps_ports for 20 Fast on bdnf: "2222 2224 2226 2228 2230 2232 2234 2236 2238 2240 2242 2246 2248 2250 2252 2254 2256 2258 2260 2262"
    # ps_ports for 10 Fast, 10 Slow on bdnf and learn: "2222 2224 2226 2228 2230 2232 2234 2236 2238 2240"
    ps_ports="2222 2224 2226 2228 2230"
#    ps_ports="2222 2224 2226 2228 2230 2232 2234 2236 2238 2240"
#    CUDA_DEVICES=("0" "1" "2" "3" "4" "0" "1" "2" "3" "4")
    CUDA_DEVICES=("0" "1" "2" "3" "4")
    bdnf_counter=0
    learn_counter=0
    erk_counter=0
    for ps_port in `echo ${ps_ports}`;
    do
        worker_port=$(($ps_port+1))

        ###### "BDNF.ISI.EDU" ######
        SERVER_CUDA_DEVICE=${CUDA_DEVICES[$bdnf_counter]}
        bdnf_counter=$((bdnf_counter + 1))
        echo "BDNF Cluster id: $bdnf_counter"

        # Init GPU TF Clusters
        ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"$SERVER_CUDA_DEVICE\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
        nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name ps --task_index 0 --port_num ${ps_port} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/ps${ps_port}.out 2>&1 &"
        ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"$SERVER_CUDA_DEVICE\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
        nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name worker --task_index 0 --port_num ${worker_port} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/worker${worker_port}.out 2>&1 &"

        # Init CPU TF Clusters
#        cpu_ps_port=$(($ps_port+100))
#        cpu_worker_port=$(($worker_port+100))
#        ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"-1\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
#        nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name ps --task_index 0 --port_num ${cpu_ps_port} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/ps${cpu_ps_port}.out 2>&1 &"
#        ssh stripeli@bdnf.isi.edu "export GRPC_VERBOSITY=DEBUG; export CUDA_VISIBLE_DEVICES=\"-1\"; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/; export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin;
#        nohup /lfs1/stripeli/metiscondaenv/bin/python3 /lfs1/stripeli/condaprojects/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name worker --task_index 0 --port_num ${cpu_worker_port} > /lfs1/stripeli/condaprojects/projectmetis/projectmetis/resources/logs/tf_servers_logs/worker${cpu_worker_port}.out 2>&1 &"

        ###### "LEARN.ISI.EDU" ######
        learn_counter=$((learn_counter + 1))
        echo "LEARN.ISI.EDU Cluster id: $learn_counter"
        ssh stripeli@learn.isi.edu "nohup /opt/stripeli/metiscondaenv/bin/python3 /opt/stripeli/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name ps --task_index 0 --port_num ${ps_port} >& /opt/stripeli/projectmetis/projectmetis/scheduling/tf_servers_logs/ps${ps_port}.out &"
        ssh stripeli@learn.isi.edu "nohup /opt/stripeli/metiscondaenv/bin/python3 /opt/stripeli/projectmetis/projectmetis/utils/init_tf_server_original.py --job_name worker --task_index 0 --port_num ${worker_port} >& /opt/stripeli/projectmetis/projectmetis/scheduling/tf_servers_logs/worker${worker_port}.out &"


        # "ERK.ISI.EDU"
#        if [[ $erk_counter -lt 5 ]]
#        then
#            erk_counter=$((erk_counter + 1))
#            echo "ERK.ISI.EDU Cluster id: $erk_counter"
#            ssh stripeli@erk.isi.edu "nohup /lfs1/anaconda/tf-cpu-conda3/bin/python3 /lfs1/metis/projectmetis/utils/init_tf_server_original.py --job_name ps --task_index 0 --port_num 3222 > /lfs1/metis/projectmetis/scheduling/tf_servers_logs/ps${ps_port}.out 2>&1 &"
#            ssh stripeli@erk.isi.edu "nohup /lfs1/anaconda/tf-cpu-conda3/bin/python3 /lfs1/metis/projectmetis/utils/init_tf_server_original.py --job_name worker --task_index 0 --port_num 3223 > /lfs1/metis/projectmetis/scheduling/tf_servers_logs/worker${worker_port}.out 2>&1 &"
#        fi

    done

fi


if [[ "$1" = "--kill" ]];
then
    ssh stripeli@bdnf.isi.edu "kill -9 \$(ps aux | grep '[s]trip' | grep '[i]nit_tf_server' | awk '{print \$2}')"
    ssh stripeli@learn.isi.edu "set processes = \`ps auwx | grep '[s]trip' | grep '[i]nit_tf_server' | awk '{print \$2}' \` && kill -9 \$processes"
#    ssh stripeli@erk.isi.edu "kill -9 \$(ps aux | grep '[s]trip' | grep '[i]nit_tf_servers.py' | awk '{print \$2}')"
fi