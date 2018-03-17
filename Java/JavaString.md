

# Class String
```java
char charAt(int index)
int codePointAt(int index)
int indexOf(String str)
int indexOf(String str, int fromIndex)
int lastIndexOf(String str)
int compareTo(String anotherString)
boolean contains(CharSequence s)
boolean startsWith(String prefix)
boolean endsWith(String suffix)
boolean matches(String regex)
int compareTo(String anotherString)
String concat(String str)
int length()
String replace(CharSequence target, CharSequence replacement)
String replaceAll(String regex, String replacement)
String replaceFirst(String regex, String replacement)
String[] split(String regex)
String substring(int beginIndex)
String substring(int beginIndex, int endIndex)
char[] toCharArray()
String toLowerCase()
String toUpperCase()
String trim()
static String valueOf(Object obj)
```

# Class StringBuilder
```java
StringBuilder append(Object obj)
StringBuilder delete(int start, int end)
StringBuilder deleteCharAt(int index)
void getChars(int srcBegin, int srcEnd, char[] dst, int dstBegin)
StringBuilder insert(int offset, Object obj)
StringBuilder replace(int start, int end, String str)
StringBuilder reverse()
void setCharAt(int index, char ch)
```

String：是不可改变的量，也就是创建后就不能在修改了。

StringBuffer：是一个可变字符串序列，它与 String 一样，在内存中保存的都是一个有序的字符串序列（char 类型的数组），不同点是 StringBuffer 对象的值都是可变的。
StringBuilder：与 StringBuffer 类基本相同，都是可变字符换字符串序列，不同点是 StringBuffer 是线程安全的，StringBuilder 是线程不安全的。

在某些特别情况下， String 对象的字符串拼接其实是被 JVM 解释成了 StringBuffer 对象的拼接，所以这些时候 String 对象的速度并不会比 StringBuffer 对象慢，而特别是以下的字符串对象生成中， String 效率是远要比 StringBuffer 快的：
```java
String S1 = “This is only a" + “ simple" + “ test";
StringBuffer Sb = new StringBuilder(“This is only a").append(“ simple").append(“ test");
```
你会很惊讶的发现，生成 String S1 对象的速度简直太快了，而这个时候 StringBuffer 居然速度上根本一点都不占优势。其实这是 JVM 的一个把戏，在 JVM 眼里，这个
```
String S1 = “This is only a" + “ simple" + “test";
```
其实就是：
```
String S1 = “This is only a simple test";
```
所以当然不需要太多的时间了。但大家这里要注意的是，如果你的字符串是来自另外的 String 对象的话，速度就没那么快了，譬如：
1 String S2 = "This is only a";
2 String S3 = "simple";
3 String S4 = "test";
4 String S1 = S2 +S3 + S4;
这时候 JVM 会规规矩矩的按照原来的方式去做。