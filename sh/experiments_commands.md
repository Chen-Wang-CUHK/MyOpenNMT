- preprocess (need set the correct file path):
```
chmod +x sh/*.sh
./sh/JsonFilePreprocess.sh
./sh/process_utils.sh
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
