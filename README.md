# AI-pontryagin（本项目）

这份 README 主要解决两件事：

1. **怎么在本项目里使用 Git（Windows + VS Code）**：暂存/提交/推送、分支、撤销。
2. **怎么正确使用 submodule（子模块）**：本项目的 `nnc/` 是一个 **Git 子模块**，它有自己独立的仓库与提交历史。

> 术语速记：
> - **工作区**：你正在编辑的文件
> - **暂存区**：`git add` 之后准备提交的文件
> - **本地提交历史**：`git commit` 之后的记录
> - **远端**：GitHub 上的仓库（例如 `origin`）

---

## 0. 当前仓库结构（重要）

- 本项目主仓库：`AI-pontryagin`
- 子模块：`nnc/`

`nnc/` 不是普通文件夹：主仓库只记录一个“指针”（nnc 当前固定在哪个提交），并不会把 nnc 的全部历史塞进主仓库里。

---

## 1. 第一次克隆（必须带 submodule）

如果你在新电脑/新目录重新拉项目，推荐用下面任一方式：

### 方式 A：一步到位（推荐）

```bash
git clone --recurse-submodules https://github.com/MinJ-YM/AI-pontryagin.git
```

### 方式 B：先 clone 再初始化 submodule

```bash
git clone https://github.com/MinJ-YM/AI-pontryagin.git
cd AI-pontryagin
git submodule update --init --recursive
```

> 如果你忘了拉 submodule：你会看到 `nnc/` 目录是空的或只有一个占位状态，此时直接执行上面的 `git submodule update --init --recursive` 就行。

---

## 2. 日常工作流（主仓库）

### 2.1 查看状态（最常用）

```bash
git status
```

你会看到：
- 哪些文件改了（modified）
- 哪些文件新建了（untracked）
- 哪些文件在暂存区（staged）

### 2.2 暂存 + 提交 + 推送

```bash
# 1) 暂存（把改动放进“待提交清单”）
git add .

# 2) 提交（写一条说明，生成一次提交记录）
git commit -m "说明你做了什么"

# 3) 推送到 GitHub（远端）
git push
```

> `git push` 推送到哪里由两件事决定：
> 1) 你当前分支跟踪哪个远端分支（例如 `origin/main`）
> 2) 你指定了什么参数（例如 `git push origin main`）

---

## 3. 子模块 nnc 的正确用法（必须懂）

### 3.1 nnc 有两套“提交/推送”

- 你改了 `nnc/` 里的文件：需要在 **nnc 仓库**里提交并推送。
- 然后还要在 **主仓库**里提交一次“子模块指针变化”。

这是 submodule 的核心：**子模块的提交不等于主仓库的提交**。

### 3.2 你修改 nnc 后如何保存到你的 nnc 远端

假设你的 `nnc` 远端是：

- `https://github.com/MinJ-YM/nnc.git`

操作步骤：

```bash
# 进入子模块目录
cd nnc

# 看子模块状态
git status

# 子模块提交
git add .
git commit -m "说明：修改了 nnc 的什么"

# 子模块推送到它自己的远端
git push

# 回到主仓库
cd ..

# 这一步非常关键：主仓库记录“nnc 指针更新到了哪个提交”
git add nnc

git commit -m "Bump nnc submodule"

git push
```

### 3.3 更新 nnc 到最新版本（拉取）

如果你只想把 nnc 更新到远端最新：

```bash
cd nnc

git pull

cd ..

git add nnc

git commit -m "Update nnc submodule"

git push
```

> 注意：如果你 `git pull` 的是你 fork 的 nnc，它会拉你 fork 的最新。

---

## 4. 远端（origin）与“推送到指定位置”

### 4.1 查看远端配置

在主仓库根目录：

```bash
git remote -v
```

你应该看到类似：

- `origin  https://github.com/MinJ-YM/AI-pontryagin.git (fetch)`
- `origin  https://github.com/MinJ-YM/AI-pontryagin.git (push)`

在子模块 `nnc/` 里也可以看：

```bash
cd nnc
git remote -v
```

### 4.2 推送到指定远端/分支

```bash
# 主仓库：推送到 origin 的 main 分支
git push origin main

# 子模块：推送到它自己的 origin 的 master/main（看子模块当前分支）
cd nnc

git push origin master
```

### 4.3 如果你想把 nnc 远端改为你自己的

在 `nnc/` 目录里：

```bash
cd nnc

git remote set-url origin https://github.com/MinJ-YM/nnc.git
```

---

## 5. 分支：为什么建议“做功能先开分支”

即使你只有自己用，也建议改大功能时新建分支：

```bash
# 创建并切换分支
git checkout -b feat/kuramoto-control

# 推送这个分支到远端
git push -u origin feat/kuramoto-control
```

回到主分支：

```bash
git checkout main
```

---

## 6. 撤销/回退（常见救命命令）

### 6.1 撤销某个文件的未提交修改

```bash
git restore path/to/file
```

### 6.2 撤销已暂存（add 过）的内容

```bash
git restore --staged path/to/file
```

### 6.3 临时把改动“收起来”（stash）

```bash
git stash push -m "临时保存"

# 恢复
git stash pop
```

---

## 7. submodule 常见坑与排查

### 7.1 clone 之后 nnc 是空的/不对

```bash
git submodule update --init --recursive
```

### 7.2 nnc 显示“detached HEAD”

子模块经常处于 detached HEAD（主仓库固定指针导致），如果你要在子模块里开发，建议：

```bash
cd nnc

# 切到一个正常分支（按你的 nnc 仓库实际分支名选择 master 或 main）
git checkout master
```

然后再修改/提交。

### 7.3 主仓库明明没改文件，但 `git status` 说 nnc 有变化

这是因为 nnc 的指针变了：

```bash
git add nnc
git commit -m "Bump nnc submodule"
```

---

## 8. 本项目运行（快速提示）

### 8.1 运行 Kuramoto 控制示例

```bash
python COC/kuramoto.py
```

如果你使用 conda：

```bash
conda activate kuramoto
python COC/kuramoto.py
```

---

## 9. 建议的 VS Code 使用方式

- 主仓库提交：在仓库根目录执行 `git add/commit/push`
- 子模块提交：打开终端 `cd nnc` 后执行 `git add/commit/push`

---

## 10. 你可以怎么提问我（最省时间）

当 Git 又“看不懂”时，直接把下面三条命令输出贴给我：

```bash
git status -sb

git remote -v

git branch -vv
```

如果是子模块问题，在 `nnc/` 目录里也贴一遍同样三条。
命令面板直接打开/切换仓库（找不到下拉框时）
Ctrl+Shift+P 打开命令面板
运行：Git: Open Repository...（中文界面可能叫“Git: 打开存储库...”）
选择 F:\AI pontryagin\nnc（或列表里名为 nnc 的那项）
然后回到 Source Control 面板提交/推送