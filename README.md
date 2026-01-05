# このリポジトリは何
ロボットインテリジェンスの課題のためのリポジトリです

# 環境構築
```
cd ~
python3 -m venv venv_robot-intelligence
cd venv_robot-intelligence
source bin/activate

# gitからインストールする場合
git clone https://github.com/nalinally/robot-intelligence.git

# zipからインストールする場合
unzip robot-intelligence.zip

cd robot-intelligence
pip install -r requirements.txt
```

# 実行の方法
```
cd ~/venv_robot-intelligence/robot-intelligence
cd scripts
python3 demo_learn.py
```
