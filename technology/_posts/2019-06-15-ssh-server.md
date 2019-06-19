---
layout: post
title: "SSH server tunnelling NAT"
description: >
    穿透NAT的SSH服务器搭建
image: /assets/img/blog/ssh.png
---
在NAT或内网防火墙下的服务器很难从外网进行访问，一些基于ssh tunnel的方法较为繁琐，而且较难维护。本文介绍使用[tmate](https://tmate.io/)这个远程终端分享工具，结合crontab定时规划和mail邮件功能来搭建一个稳定可用的SSH服务器。

本文主要在Ubuntu和Mac环境测试可用。

# 背景
实验室的服务器配置在内网环境下，在外网无法访问。这使实验开展、模型训练测试有较多不便。考虑搭建一个简便的框架使得外网的SSH连接成为可能。

# tmate
tmate是一个开源项目，提供高效终端共享的功能，可以类比为终端版本的Teamviewer。

## 安装
Ubuntu输入，可能会有网络连接上的一些问题，可自行解决
```
sudo apt-get install software-properties-common && \
sudo add-apt-repository ppa:tmate.io/archive    && \
sudo apt-get update                             && \
sudo apt-get install tmate
```
Mac输入
```
brew install tmate
```

## 使用
在使用前，需要已创建ssh私钥公钥，可通过`ssh-keygen`自行解决。

直接输入`tmate`能够启动共享的终端，`tmate show-messages`可以得到相关的ssh与web访问地址。

直接开启终端的一个问题是窗口关闭会直接关闭ssh通道，这无法解决外网访问的问题。需要使tmate在后台运行。
```
tmate -S /tmp/tmate.sock new-session -d
tmate -S /tmp/tmate.sock wait tmate-ready
tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}'
tmate -S /tmp/tmate.sock display -p '#{tmate_web}'
```
能够实现后台运行的功能。

## 监控
到上一步骤，我们已经可以从外网使用ssh连接内网的服务器了。但上面后台运行的tmate服务器可能会出现奔溃、失联等情形。需要有监控的手段，来维护始终能有一个相应的服务器正常运行。这里使用crontab设置定时任务进行监控。

创建文件`ssh.sh`，包含以下内容
```
#!/bin/bash
tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}' > /Users/ben/ssh.txt 
var=$(cat /Users/ben/ssh.txt)
if [ ! -n "$var" ]; then
    tmate -S /tmp/tmate.sock kill-session
    tmate -S /tmp/tmate.sock new-session -d              
    tmate -S /tmp/tmate.sock wait tmate-ready            
    tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}' > /Users/ben/ssh.txt  
    mail -s "SSH reactivation" xxx@xxx.com < /Users/ben/ssh.txt
fi
```
它监控是否有运行中的服务器，若没有则设立一个新服务器，并通过邮件将新的ssh连接代码发送至个人邮箱。文件保存后，需要将其权限修改为可执行`chmod a+x ssh.sh`。

当然，发送邮件需要smtp的设置。输入`sudo vim /etc/mail.rc`，在文件后附加发件人的相关信息，如
```
set from=xxx@163.com
set smtp=smtp.163.com
set smtp-auth-user=xxx@163.com
set smtp-auth-password=xxx
set smtp-auth=login
```

随后，可以设置定时监测了。输入`crontab -e`，添加自动执行的命令
```
*/30 * * * * . /etc/profile; /Users/ben/ssh.sh &> /Users/ben/tmp.log
```
表示每30分钟运行`ssh.sh`，即进行服务器运行状况的判断。这里需要注意`PATH`的设定，代码中添加了当前用户的环境设定。

保存后，`sudo service cron restart`重新启动服务来开始监控，MAC下不需要重新启动这一操作。

## 国内服务器
国外服务器`tmate.io`非常不稳定，考虑迁移到国内的服务器来来做转发工作，这里使用阿里云的ECS服务。

启动Ubuntu后，首先安装相关的依赖包，再之前进行更新操作
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git-core build-essential pkg-config libtool libevent-dev libncurses-dev zlib1g-dev automake libssh-dev cmake ruby
```
随后git下载所需的源码。
```
git clone https://github.com/tmate-io/tmate-slave.git && cd tmate-slave
```
为了生成rsa及ecdsa公钥。这里需要提前将`./create_key.sh`修改，将原本生成的`ed25519`改成`ecdsa`，具体的，将该文件最后行改为
```
gen_key rsa && gen_key ecdsa || exit 1
```
随后，可以生成公钥
```
./create_keys.sh 
./autogen.sh && ./configure && make
sudo ./tmate-slave
```
对于生成的公钥，需要记录其md5值
```
ssh-keygen -E md5 -lf keys/ssh_host_rsa_key.pub
ssh-keygen -E md5 -lf keys/ssh_host_ecdsa_key.pub
```
完成安装后，开启服务器，监控9999端口
```
sudo ./tmate-slave -k /root/tmate-slave/keys -p 9999 -h [替换为公网IP地址] -v
```
需要登陆阿里云的安全组规则，将9999端口新增入方向的允许策略

在客户端，即需要共享终端的上，需要修改tmate的配置
```
vim ~/.tmate.conf
```
新建文件，随后添加
```
set -g tmate-server-host "[替换为公网IP地址]"
set -g tmate-server-port 9999
set -g tmate-server-rsa-fingerprint   "[aa:bb:cc替换为rsa的md5值]"
set -g tmate-server-ecdsa-fingerprint "[aa:bb:cc替换为ecdsa的md5值]"
set -g tmate-identity ""
```
设置完成后，可以通过`tmate`进行调试

### 服务器安装过程错误
* msgpack
`configure: error: "msgpack >= 1.2.0 not found"`
需要手动编译安装静态包
```
git clone https://github.com/msgpack/msgpack-c.git 
cd msgpack-c
cmake .
make
make install
```
* libssh
`configure: error: "libssh >= 0.7.0 not found"`
需要手动编译安装静态包
```
git clone https://git.libssh.org/projects/libssh.git libssh
cd libssh
mkdir build
cmake ..
make
make install
```
* libssh.so.4
`./tmate-slave: /usr/lib/x86_64-linux-gnu/libssh.so.4: no version information available (required by ./tmate-slave)`
需要将已安装的其他版本libssh卸载
```
dpkg -l|grep libssh
 apt-get remove libssh-4
```

到这里，通过tmate, crontab和mail来搭建穿通NAT的稳定的SSH服务器就基本完成了。