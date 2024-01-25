


from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# 加载模型和分词器
model_name = "vis-Mistral-7B-v0.1-ChartDataset-to-PresentationScript"
model_path = "vis-Mistral-7B-v0.1-ChartDataset-to-PresentationScript"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# @app.route('/generate', methods=['POST'])
# def generate_text():
#     data = request.json
#     input_text = data['text']
    
#     # 确保输入长度适合模型的限制
#     inputs = tokenizer(
#         input_text, 
#         return_tensors='pt', 
#         padding=True, 
#         truncation=True, 
#         max_length=5000  # 或模型支持的其他最大长度
#     )

#     # 生成文本
#     outputs = model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_new_tokens=5000  # 控制生成的最大新令牌数
#     )
#     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'generated_text': response_text})

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get("inputs", "")

    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # 参数设置
    params = data.get("parameters", {})
    temperature = params.get("temperature", 0.2)
    top_p = params.get("top_p", 0.95)
    max_new_tokens = params.get("max_new_tokens", 256)
    do_sample = params.get("do_sample", True)

    # 分词处理
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

    # 生成文本
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
    )

    # 解码文本
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': response_text})

if __name__ == '__main__':
    app.run(debug=True)
