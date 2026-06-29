# GitHub 推送指南

## 方法一：通过 GitHub CLI（推荐）

### 1. 安装 GitHub CLI
```bash
# macOS
brew install gh

# Ubuntu/Debian
sudo apt install gh

# Windows
winget install --id GitHub.cli
```

### 2. 登录 GitHub
```bash
gh auth login
# 按提示选择 HTTPS / Y / 粘贴token
```

### 3. 创建仓库并推送
```bash
cd /path/to/etf_size_curse_research
gh repo create etf-size-curse-research --public --source=. --push
```

## 方法二：通过浏览器 + Git 命令

### 1. 在 GitHub 上创建新仓库
- 访问 https://github.com/new
- 仓库名: `etf-size-curse-research`
- 选择 Public
- 不要勾选 "Initialize this repository with a README"
- 点击 "Create repository"

### 2. 推送本地代码
```bash
cd /path/to/etf_size_curse_research
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/etf-size-curse-research.git
git push -u origin main
```

### 3. 输入用户名和密码（或Token）
- 用户名: 你的 GitHub 用户名
- 密码: 使用 Personal Access Token（Settings -> Developer settings -> Personal access tokens）

## 方法三：使用 GitHub Desktop

1. 打开 GitHub Desktop
2. File -> Add local repository -> 选择项目文件夹
3. 点击 "Publish repository"
4. 选择 Public，点击 "Publish"

---

推送成功后，你的项目将可在以下地址访问：
```
https://github.com/YOUR_USERNAME/etf-size-curse-research
```
