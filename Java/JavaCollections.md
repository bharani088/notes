![](http://4.bp.blogspot.com/-DvsfKh9clI0/UU3sK7J17jI/AAAAAAAAARU/VnHJDjImzw4/s1600/java-collection-hierarchy.png)

![](https://dzone.com/storage/temp/1821399-sortedmap.png)

# class Arrays
```
static List<T> asList(T... a)
static int binarySearch(Object[] a, Object key)
static boolean[] copyOf(int[] original, int newLength)
static boolean[] copyOfRange(int[] original, int from, int to)
static boolean equals(int[] a, int[] a2)
static void fill(int[] a, int val)
static void sort(int[] a)
```

# class Collections
```
static <T> boolean addAll(Collection<? super T> c, T... elements)
static <T> int binarySearch(List<? extends Comparable<? super T>> list, T key)
static <T> void copy(List<? super T> dest, List<? extends T> src)
static <T> void fill(List<? super T> list, T obj)
static int frequency(Collection<?> c, Object o)
static int indexOfSubList(List<?> source, List<?> target)
static <T extends Object & Comparable<? super T>> T max(Collection<? extends T> coll)
static <T extends Object & Comparable<? super T>> T min(Collection<? extends T> coll)
static void reverse(List<?> list)
static <T extends Comparable<? super T>> void sort(List<T> list)
static void swap(List<?> list, int i, int j)
```

# Interface Collection<E>
```  
boolean add(E e)
boolean addAll(Collection<? extends E> c)
void clear()
boolean contains(Object o)
boolean containsAll(Collection<?> c)
boolean isEmpty()
boolean remove(Object o)
boolean removeAll(Collection<?> c)
boolean retainAll(Collection<?> c)
int size()
<T> T[] toArray(T[] a)
```

# Interface List<E>
```
void add(int index, E element)
E get(int index)
int indexOf(Object o)
Iterator<E> iterator()
ListIterator<E> listIterator()
E remove(int index)
E set(int index, E element)
default void sort(Comparator<? super E> c)
List<E> subList(int fromIndex, int toIndex)
```

## Class ArrayList<E>
```
protected void removeRange(int fromIndex, int toIndex)
```

## Class LinkedList<E>
```
boolean offer(E e) // tail<— ,==add==addLast
E poll() // <—head ,==removeFirst
void push(E e) // —>head ,==addFirst
E pop() // <—head ,==removeFirst==poll
E peek()
```

# Interface Set<E>

## Class HashSet<E>
## Class TreeSet<E>
```
E first()
E last()
SortedSet<E> subSet(E fromElement, E toElement)
```

# Interface Queue<E>
```
boolean offer(E e) // tail<—
E poll() // <—head
E peek()
```
## Class PriorityQueue<E>

# Interface Map<K,V>
```
V put(K key, V value)
V get(Object key)
V remove(Object key)
default V replace(K key, V value)
boolean containsKey(Object key)
boolean containsValue(Object value)
Set<K> keySet()
Set<Map.Entry<K,V>> entrySet()
int size()
Collection<V> values()
```
## Class HashMap<K,V>
## Class TreeMap<K,V>

==========================

# Compare

## HashSet
如果两个对象通过equals()方法比较返回true，但这两个对象的hashCode()方法返回不同的hashCode值时，这将导致HashSet会把这两个对象保存在Hash表的不同位置，从而使对象可以添加成功，这就与Set集合的规则有些出入了。
所以，我们要明确的是: equals()决定是否可以加入HashSet、而hashCode()决定存放的位置，它们两者必须同时满足才能允许一个新元素加入HashSet。
但是要注意的是: 如果两个对象的hashCode相同，但是它们的equlas返回值不同，HashSet会在这个位置用链式结构来保存多个对象。而HashSet访问集合元素时也是根据元素的HashCode值来快速定位的，这种链式结构会导致性能下降。
所以如果需要把某个类的对象保存到HashSet集合中，我们在重写这个类的equlas()方法和hashCode()方法时，应该尽量保证两个对象通过equals()方法比较返回true时，它们的hashCode()方法返回值也相等。

## ArrayList
如果一开始就知道ArrayList集合需要保存多少元素，则可以在创建它们时就指定initialCapacity大小，这样可以减少重新分配的次数。

## LinkedList
同时表现双端队列、栈的用法。

## PriorityQueue
不允许插入null元素，它还需要对队列元素进行排序，PriorityQueue的元素有两种排序方式：自然排序（集合元素实现Comparable接口）、定制排序（创建PriorityQueue队列时传入一个Comparator对象）。

## HashMap/Hashtable
当使用自定义类作为HashMap、Hashtable的key时，如果重写该类的equals(Object obj)和hashCode()方法，则应该保证两个方法的判断标准一致--当两个key通过equals()方法比较返回true时，两个key的hashCode()的返回值也应该相同。

Set和Map的关系十分密切，java源码就是先实现了HashMap、TreeMap等集合，然后通过包装一个所有的value都为null的Map集合实现了Set集合类


# 对 List 的选择
1、对于随机查询与迭代遍历操作，数组比所有的容器都要快。所以在随机访问中一般使用 ArrayList
2、LinkedList 使用双向链表对元素的增加和删除提供了非常好的支持，而 ArrayList 执行增加和删除元素需要进行元素位移。
3、对于 Vector 而已，我们一般都是避免使用。
4、将 ArrayList 当做首选，毕竟对于集合元素而已我们都是进行遍历，只有当程序的性能因为 List 的频繁插入和删除而降低时，再考虑 LinkedList。

# 对 Set 的选择
1、HashSet 由于使用 HashCode 实现，所以在某种程度上来说它的性能永远比 TreeSet 要好，尤其是进行增加和查找操作。
3、虽然 TreeSet 没有 HashSet 性能好，但是由于它可以维持元素的排序，所以它还是存在用武之地的。

# 对 Map 的选择
1、HashMap 与 HashSet 同样，支持快速查询。虽然 HashTable 速度的速度也不慢，但是在 HashMap 面前还是稍微慢了些，所以 HashMap 在查询方面可以取代 HashTable。
2、由于 TreeMap 需要维持内部元素的顺序，所以它通常要比 HashMap 和 HashTable 慢。