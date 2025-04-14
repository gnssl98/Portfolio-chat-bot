from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 모델 로딩
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 테스트
text = "당신의 기술 스택은 무엇인가요?"
translated = translate(text, "ko_KR", "en_XX")
print("번역 결과:", translated)
