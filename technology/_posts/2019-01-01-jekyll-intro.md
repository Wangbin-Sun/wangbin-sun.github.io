---
layout: post
title: "Jekyll Introduction"
description: >
    Jekyll是该博客使用的前端框架，本文将对其初步介绍
image: /assets/img/blog/jekyll.png
---
之前在Github上搜索创建网站的教程，发现Jekyll是推荐的框架，于是简单研究了一下。

# 背景
[Jekyll](https://jekyllrb.com/)是Github Pages默认支持的一种前端框架。官网介绍其特点包括：**简单**，不需要数据库后端、评论功能、调节功能、系统更新等繁琐操作，只需要输入文本内容即可；**静态**，支持Markdown、Liquid、HTML&CSS，静态站点部署所见即所得；**博客友好**，永久链接、分类、页面、博文、自定义布局等是框架的核心关注点。
# 安装
直接使用gem安装bundler和jekyll
```
gem install bundler jekyll
```

# 运行
需自行创建一个blog
```
jekyll new blog
```
或直接从template中，clone一个模版
```
cd blog
bundle exec jekyll serve
```
随后在[http://localhost:4000](http://localhost:4000)访问即可

# 示例
以[Hydejack](https://hydejack.com/)为例，git clone之后，`cd`至对应的文件夹，查看`Gemfile.lock`，发现其bundle版本为1.16.1，下载对应班对的bundler
```
gem install bundler -v 1.16.1
```
并考虑卸载当前版本的bundler，通过`bundle -v`查看版本，本机版本是1.17.3
```
gem uninstall bundler -v 1.17.3
```
现在再使用`bundle -v`，发现版本一致，下载所需的gems包
```
bundle install
```
得到成功的消息后，在本地运行服务器
```
bundle exec jekyll server
```
访问[http://localhost:4000](http://localhost:4000)可看到示例效果。

后续的定制化可以修改相应的`_config.yml`等等
