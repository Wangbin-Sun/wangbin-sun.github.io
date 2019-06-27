---
layout: post
title: "Git Command Summary"
description: >
    git工具的使用总结
image: /assets/img/blog/git.png
---
git是代码管理、团队协作非常重要的工具，本文总结其较为常见的用法。

# 基础命令
* git init 创建仓库
* git add A 选择更新文件
* git commit -m "A" 提交更新文件，注释A
* git commit --amend 修改上一次commit的内容
* git status 当前仓库状态
* git diff A 比较A前后版本变化情况
* git log [--pretty=oneline]  [--graph] [--abbrev-commit]呈现历史版本变更记录，［呈现单行］/ [呈现分支合并图] [缩写commit编号]
* git reset --hard HEAD^ 回滚到上一版本（HEAD~i，回滚到上i个版本）
* git reflog 包括被回滚过的所有版本纪录
* git checkout -- A 使文件A恢复成最近一次commit或add的内容
* git reset HEAD A 把放入缓存区待提交的A文件拿出来
* git rm A 准备删除文件
* git remote add origin git@github.com:Username/Reponame.git 关联到远程仓库[需要事先创建]
* git push [-u] origin master 将本地master推送到远程origin，［首次推送需关联 - u］
* git push --force-with-lease origin master 将修改后的commit log强行推送到远程
* git clone git@github.com:Username/Reponame.git 从远程克隆到本地
* git checkout -b A 创建并切换到一个新分支A，（-b相当于git branch+checkout A)
* git checkout -b dev origin/dev 克隆远程的dev分支到本地，并在本地创建dev
* git branch 查看当前分支
* git checkout A 切换到分支A
* git merge [--no-ff] [-m "B"] A 将分支A与当前分支合并 [禁用fast forward，保留合并记录，推荐] / [注释B]
* git branch [-d] [-D] A 删除分支A/[D强行删除未合并的分支]
* git stash 临时储存工作区，文件滚回HEAD
* git stash list 保存的工作现场列表
* git stash apply 恢复临时工作区
* git stash drop 删除临时信息
* git stash pop 功能等于apply+drop
* git remote -v 显示远程仓库的推送地址
* git branch --set-upstream-to=origin/dev dev 将本地的dev与远程仓库dev分支关联
* git pull 抓取最近的提交，试图合并（强行抓取先stash，随后pull，在stash drop/或者先checkout -- . 再pull）
* git rebase 修改记录中的分叉成直线
* git tag A 给当前分支准备的commit打标签
* git tag A f52c633 给特定的commit打标签
* git show A 显示标签A的提交
* git tag -a v0.1 -m A f52c633 给特定的commit打标签，并辅以注释A
* git tag -d A 删除标签A
* git push origin v1.0 /[--tag] 推送标签[一次性推送全部尚未推送到远程的本地标签]
* git push origin :ref/tags/v0.9 删除已推送到远程的标签，需现在本地删除
* git remote rm origin 删除远程关联

# 一些combo

## commit后回滚操作
git log 查看需要回滚的commit记录  
git reset --hard xxx 将本机代码回滚到xxx commit后的记录  
git push -f 将本机代码推送到云端  
该方式会将已有的记录完全删除，且无法撤销

## commit前回滚操作
git checkout -- . 所有文件均回滚

## 下载当前分支有修改的文件
直接git pull报错，需要先将当前工作区备份  
git stash 备份  
git pull 下载文件  
git stash pop 当备份信息释放应用