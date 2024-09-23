from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

# 指明预训练模型
model_name = 'liam168/trans-opus-mt-zh-en'
# 加载预训练模型
model = AutoModelWithLMHead.from_pretrained(model_name)
# 加载词嵌入层
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 使用管道的方式进行机器翻译
translation = pipeline('translation_zh_to_en', model=model, tokenizer=tokenizer)
# 将要翻译的文本传递到API中
out = translation('你好，很高兴认识你！', max_length=400)
print(out)