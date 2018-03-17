> [https://jaywcjlove.github.io/linux-command/](https://jaywcjlove.github.io/linux-command/)

# Dir & File Management
==================

## DIR

### tree
以树状图列出目录的内容
>tree /private/ -L 1

### rmdir
删除空目录
>rmdir -p bin/os_1

### mkdir
创建目录
>mkdir -p-m 750 bin/os_1

### rm
删除一个目录中的一个或多个文件或目录，也可以将某个目录及其下属的所有文件及其子目录均删除掉
>rm -i test example
rm -r *

### pwd
以绝对路径的方式显示用户当前工作目录

### cd
切换工作目录

### ls
ls ### 仅列出当前目录可见文件
ls -l ### 或ll，列出当前目录可见文件详细信息
ls -hl ### 列出详细信息并以可读大小显示文件大小
ls -al ### 列出所有文件（包括隐藏）的详细信息

### mv
用来对文件或目录重新命名，或者将文件从一个目录移到另一个目录中

## FILE

### cp
用来将一个或多个源文件或者目录复制到指定的目的文件或目录

### cat
连接文件并打印到标准输出设备上
>cat m1 （在屏幕上显示文件ml的内容）
cat m1 m2 （同时显示文件ml和m2的内容） 
cat m1 m2 > file （将文件ml和m2合并后放入文件file中）

### chmod
用来变更文件或目录的权限
>chmod u+x,g+w f01　　### 为文件f01设置自己可以执行，组员可以写入的权限
chmod u=rwx,g=rw,o=r f01 
chmod 764 f01 chmod a+x f01　　 ### 对文件f01的u,g,o都设置可执行属性

### chown
用来变更文件或目录的拥有者或所属群组

# Sys
### chkconfig
检查、设置系统的各种服务
>chkconfig --list             ###列出所有的系统服务。
chkconfig --add httpd        ###增加httpd服务。
chkconfig --del httpd        ###删除httpd服务。
chkconfig --level httpd 2345 on        ###设置httpd在运行级别为2、3、4、5的情况下都是on（开启）的状态。
chkconfig --list mysqld        ### 列出mysqld服务设置情况。
chkconfig --level 35 mysqld on ### 设定mysqld在等级3和5为开机运行服务，--level 35表示操作只在等级3和5执行，on表示启动，off表示关闭。
chkconfig mysqld on            ### 设定mysqld在各等级为on，“各等级”包括2、3、4、5等级。
chkconfig –level redis 2345 on ### 把redis在运行级别为2、3、4、5的情况下都是on（开启）的状态。

### iptables
Linux上常用的防火墙软件