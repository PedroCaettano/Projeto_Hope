# Baseado no video da Danki Code no YouTube

import sys
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Config,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Função para gerar resposta com token de parada
def generate_with_stop_token(model, input_text, tokenizer, stop_token='<|endoftext|>', temperature=0.1, max_length=50):
    eos_token_id = tokenizer.encode(stop_token)[0]
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    attention_mask = input_tokens['input_ids'].ne(tokenizer.pad_token_id).float()

    output_tokens = model.generate(
        input_ids=input_tokens['input_ids'],
        attention_mask=attention_mask,
        temperature=temperature,
        max_length=max_length,
        eos_token_id=eos_token_id,
        repetition_penalty=1.5  # Ajuste este valor para controlar a penalização da repetição
    )

    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return answer

# Função para remover caracteres indesejados da resposta
def remover_caracteres(texto: str, caracteres: list = ['</s>', '<s>', '</s']):
    for caractere in caracteres:
        texto = texto.replace(caractere, '')
    return texto

# Carregar o modelo e o Tokenizer
config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# Função para treinar o modelo
def train_model(model, tokenizer, train_texts_path):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_texts_path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

# Testar o modelo
if __name__ == "__main__":
    train_texts_path = "path/to/your/train_dataset.txt"  # Substitua pelo caminho correto do seu dataset de treino
    train_model(model, tokenizer, train_texts_path)

    while True:
        input_text = input("Digite a entrada: ")
        answer = generate_with_stop_token(model, input_text, tokenizer, temperature=1.0)
        print("Resposta:", remover_caracteres(answer))
