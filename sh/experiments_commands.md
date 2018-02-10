- preprocess json file into .txt file:
```
chmod +x sh/*.sh
./sh/JsonFilePreprocess.sh
```
- extract answer into another .txt file:
```
chmod +x sh/*.sh
./sh/process_utils.sh
```
- preprocess .txt file, combine into .pt file and build vocabs:
```
chmod +x sh/*.sh
./sh/preprocess.sh
```
- use glove embedding:
```
chmod +x sh/*.sh
./sh/preprocessing_embedding_pytorch.sh
```
- train:
```
chmod +x sh/*.sh
./sh/train.sh
```
- generate:
choose which model to use, and modify sh/translate.sh
```
chmod +x sh/*.sh
./sh/translate.sh
```
