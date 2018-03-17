# MongoDB 
运行方式主要基于两个概念：集合（collection）与文档（document）。

## 数据库

数据库是集合的实际容器。每一数据库都在文件系统中有自己的一组文件。一个 MongoDB 服务器通常有多个数据库。

## 集合

集合就是一组 MongoDB 文档。它相当于关系型数据库（RDBMS）中的表这种概念。集合位于单独的一个数据库中。集合不能执行模式（schema）。一个集合内的多个文档可以有多个不同的字段。一般来说，集合中的文档都有着相同或相关的目的。

## 文档

文档就是一组键-值对。文档有着动态的模式，这意味着同一集合内的文档不需要具有同样的字段或结构。

下表展示了关系型数据库与 MongoDB 在术语上的对比：

| 关系型数据库 | MongoDB
| --- | ---
| 数据库 | 数据库
| 表 | 集合
| 行 | 文档
| 列 | 字段
| 表Join | 内嵌文档
| 主键 | 主键（由 MongoDB 提供的默认 key_id）


# MongoDB Shell (mongo)

## Start mongo

Once you have installed and have started MongoDB, connect the mongo shell to your running MongoDB instance. Ensure that MongoDB is running before attempting to launch the mongo shell:
mongo

## Help in mongo Shell

Type help in the mongo shell for a list of available commands and their descriptions:

help

用 use + 数据库名称 的方式来创建数据库

db.dropDatabase() 命令用于删除已有数据库

>use mydb
switched to db mydb
>db.dropDatabase()
>{ "dropped" : "mydb", "ok" : 1 }
>

创建集合采用 db.createCollection(name, options) 方法
db.collection.drop() 来删除数据库中的集合。

## 数据类型

MongoDB 支持如下数据类型：

* String：字符串。存储数据常用的数据类型。在 MongoDB 中，UTF-8 编码的字符串才是合法的。
* Integer：整型数值。用于存储数值。根据你所采用的服务器，可分为 32 位或 64 位。
* Boolean：布尔值。用于存储布尔值（真/假）。
* Double：双精度浮点值。用于存储浮点值。
* Min/Max keys：将一个值与 BSON（二进制的 JSON）元素的最低值和最高值相对比。
* Arrays：用于将数组或列表或多个值存储为一个键。
* Timestamp：时间戳。记录文档修改或添加的具体时间。
* Object：用于内嵌文档。
* Null：用于创建空值。
* Symbol：符号。该数据类型基本上等同于字符串类型，但不同的是，它一般用于采用特殊符号类型的语言。
* Date：日期时间。用 UNIX 时间格式来存储当前日期或时间。你可以指定自己的日期时间：创建 Date 对象，传入年月日信息。
* Object ID：对象 ID。用于创建文档的 ID。
* Binary Data：二进制数据。用于存储二进制数据。
* Code：代码类型。用于在文档中存储 JavaScript 代码。
* Regular expression：正则表达式类型。用于存储正则表达式。

## Insert a Document
db.COLLECTION_NAME.insert({THE_JSON_OBJECT})
If the document passed to the insert() method does not contain the `_id` field, the mongo shell automatically adds the field to the document and sets the field’s value to a generated ObjectId.

## Query for All Documents in a Collection
To return all documents in a collection, call the find() method without a criteria document.
db.restaurants.find()




