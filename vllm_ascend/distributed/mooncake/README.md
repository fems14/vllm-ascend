



export PYTHONPATH=$PYTHONPATH:/home/xxxx/vllm
export PYTHONPATH=$PYTHONPATH:/home/xxxx/vllm-ascend
export MMC_LOCAL_CONFIG_PATH=/home/xxxx/config_npu0.conf
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=2,3
# export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="/home/d00838720/ranktable_84.json"
# --tensor-parallel-size 2\
# --data-parallel-size 2 \
#     --data-parallel-size-local 2 \
#     --data-parallel-address 141.61.33.167 \
#     --data-parallel-rpc-port 9200 \
# --tensor-parallel-size 2\
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/weight/Qwen3-8B \
    --port 8200 \
    --max-model-len 10000 \
    --tensor-parallel-size 2\
    --data-parallel-size 1 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorStoreV1","kv_role":"kv_consumer","kv_connector_extra_config":{"use_layerwise": false}}' > log_c.log 2>&1



export PYTHONPATH=$PYTHONPATH:/home/xxxx/vllm
export PYTHONPATH=$PYTHONPATH:/home/xxxx/vllm-ascend
export MMC_LOCAL_CONFIG_PATH=/home/xxxx/config_npu0.conf
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
# export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="/home/d00838720/ranktable_84.json"
# --tensor-parallel-size 2\
    # --data-parallel-size 2 \
    # --data-parallel-size-local 2 \
    # --data-parallel-address 141.61.33.167 \
    # --data-parallel-rpc-port 9100 \
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/weight/Qwen3-8B \
    --port 8100 \
    --max-model-len 10000 \
    --enforce-eager \
    --tensor-parallel-size 2\
    --data-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorStoreV1","kv_role":"kv_producer","kv_connector_extra_config":{"use_layerwise": false}}' > log_p.log 2>&1






cd /home/f30058701/vllm
python3 examples/online_serving/disaggregated_serving/disagg_proxy_demo.py --model /mnt/weight/Qwen3-8B --prefill 90.90.97.27:8100 --decode 90.90.97.27:8200 --port 8000




# meta service启动url
# 在meta service非HA场景，请与mmc-meta.conf中的配置项保持一致
# 在K8S集群meta service主备高可用场景，请配置为ClusterIP地址
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
# 日志级别
ock.mmc.log_level = info

# TLS安全通信证书相关配置
ock.mmc.tls.enable = false
ock.mmc.tls.top.path = /opt/ock/security/
ock.mmc.tls.ca.path = certs/ca.cert.pem
ock.mmc.tls.ca.crl.path = certs/ca.crl.pem
ock.mmc.tls.cert.path = certs/client.cert.pem
ock.mmc.tls.key.path = certs/client.private.key.pem
ock.mmc.tls.key.pass.path = certs/client.passphrase
ock.mmc.tls.package.path = /opt/ock/security/libs/

# client的总数
ock.mmc.local_service.world_size = 4
# BM服务启动url，在K8S集群meta service主备高可用场景，在Pod启动时自动修改为PodIP
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
# ip需要设为RDAM网卡ip，可以使用show_gids命令查询
ock.mmc.local_service.hcom_url = tcp://127.0.0.1:7000
# 数据传输协议，DRAM池使用roce，HBM池使用sdma
ock.mmc.local_service.protocol = sdma
# DRAM空间使用量，单位字节，默认128MB，和HBM二选一，需要2M对齐
ock.mmc.local_service.dram.size = 0
# HBM空间使用量，单位字节，和DRAM二选一
ock.mmc.local_service.hbm.size = 209715200

ock.mmc.client.timeout.seconds = 60



#### 运行
**1. 启动metaservice独立进程**
```
export MMC_META_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-meta.conf
cd /usr/local/mxc/memfabric_hybrid/1.0.0/aarch64-linux/bin/
./mmc_meta_service
```

**2. 通过pymmc提供的接口初始化客户端并拉起localservice，执行数据写入、查询、获取、删除等**
```
export MMC_LOCAL_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-local.conf
python3 -m unittest test_mmc_demo.py
python3 -m unittest test_mmc_layer.py
```





curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/mnt/weight/Qwen3-8B",
  "prompt": "Hello. I have a question. The president of the United States is",
  "max_tokens": 200,
  "temperature":0.0
}'




curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
        "model": "/mnt/weight/Qwen3-8B",
        "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?",
        "max_tokens": 256,
        "temperature":0.0
}'
