##池化场景1p1d运行指南

#### 1、配置mooncake.json

环境变量`MOONCAKE_CONFIG_PATH`配置为mooncake.json

```
{
    "local_hostname": "xx.xx.xx.xx",
    "metadata_server": "http://xx.xx.xx.xx:8088/metadata",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088"
}
```

local\_hostname:配置为当前主节点的ip地址，
metadata\_server: 服务端所在ip与端口，
protocol: 配置为ascend使用mooncake的hccl通信,
device\_name: ""
master\_server\_address：配置master服务的ip和port

#### 2、启动metadata\_server

在mooncake文件夹中：

```
cd mooncake-transfer-engine/example/http-metadata-server
go run . --addr=:8088
```

#### 3、启动mooncake\_master

在mooncake文件夹下：

```
mooncake_master --port 50088
```

#### 4、启动p节点

bash store\_producer.sh，store\_producer.sh中内容：

```
export MOONCAKE_CONFIG_PATH="/xx/xx/mooncake.json"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0
python3 -m vllm.entrypoints.openai.api_server \
    --model /xx/xx/modle/Qwen2.5-7B-Instruct \
    --port 8100 \
    --max-model-len 10000 \
    --block-size 8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_buffer_device": "npu","kv_rank": 0, "kv_connector_extra_config":{"use_layerwise": true}}' > log_p.log

```

use\_layerwise控制是否开分层

#### 5、启动d节点

bash store\_consumer.sh， store\_consumer.sh中内容：

```
export MOONCAKE_CONFIG_PATH="/xx/xx/mooncake.json"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=4
python3 -m vllm.entrypoints.openai.api_server \
    --model /xx/xx/modle/Qwen2.5-7B-Instruct \
    --port 8200 \
    --max-model-len 10000 \
    --block-size 8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_buffer_device": "npu","kv_rank": 1, "kv_connector_extra_config":{"use_layerwise": true}}' > log_c.log

```

#### 6、启动proxy

在vllm下执行：

```
python3 examples/online_serving/disaggregated_serving/disagg_proxy_demo.py --model /xx/xx/modle/Qwen2.5-7B-Instruct --prefill 141.61.41.84:8100 --decode 141.61.41.84:8200 --port 8000
```

#### 7、下发推理请求：

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/xx/xx/modle/Qwen2.5-7B-Instruct",
  "prompt": "Hello. I have a question. The president of the United States is",  
  "max_tokens":200                                                              
  }'

```

#### 提示

上述所有步骤中的文件路径需写成文件实际路径，模型也要配置为模型实际所在路径
