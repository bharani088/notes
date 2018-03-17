"ls -l” or “ll"

![](https://www.linuxtrainingacademy.com/wp-content/uploads/2017/02/linux-permissions-chart.png)

Each file and directory has three user based permission groups:

* owner: The Owner permissions apply only the owner of the file or directory, they will not impact the actions of other users.
* group: The Group permissions apply only to the group that has been assigned to the file or directory, they will not effect the actions of other users.
* all users: The All Users permissions apply to all other users on the system, this is the permission group that you want to watch the most.

The Permission Types that are used are:
* r: Read
* w: Write
* x: Execute

The numbers are a binary representation of the rwx string.
* r = 4
* w = 2
* x = 1

