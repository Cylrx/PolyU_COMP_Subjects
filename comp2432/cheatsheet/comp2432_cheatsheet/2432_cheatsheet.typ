#import "@preview/lovelace:0.3.0": *

#set page(
  paper: "a4",
  margin: (
    top: 0in,
    bottom: 0in,
    left: 0in,
    right: 0in,
  )
)
#set par(justify: true)
#set text(
  size: 6.5pt,
)

#let sps = {v(2pt)}
#let dd(y, x) = {$(diff #y) / (diff #x)$}
#let ddx(y) = {$(diff #y)/(diff x)$}
#let big(x) = {$upright(bold(#x))$}
#let code(c) = {highlight(raw(c), fill: silver)}
#let head(h) = {text(fill: maroon, weight: "black", style: "oblique")[#h]}
#let subhead(h) = {text(fill: blue, weight: "black", style: "oblique")[#h]}
#let def(x, y, g: 1em) = {grid(columns: 2, gutter: g, [#set text(style: "italic"); #x], [#y])}

#columns(3, gutter: 6pt)[
#highlight[*Name*: WANG Yuqi *Student ID*: 23110134D]

#head([Introduction])

#subhead([OS Components])\
#place(right, float: false, image("layers.png", width: 14%))
*Level 1*: Kernel\
- Process scheduling, memory allocation, syscall ...
*Level 2*: Syscalls\
- _Interface_ between user and applications and kernel
- File I/O: `open()`, `write()` ... 
- Process Control: `exec()`, `exit()`, `fork()` ...
- Communication: `socket()`, `send()`, `recv()` ...

*Level 3*: Shell & System Programs\
- Shell: CLI. Translates commands $->$ syscalls
- System Programs: Apps built on syscalls. (compilers, disk formatter)


#subhead([OS Functions])\
*Process Management*: Create, delete, suspend, resume, schedule\
- _Multiprogramming_: Multiple programs loaded in RAM at the same time (switch when I/O bound)
- _Multitasking_: Rapidly switch between processes on time slice (illusion of simutaneous execution)
*Memory Management*: Register < Cache < RAM < 2nd Storage
- Tasks: Allocate / Deallocate, Protect unauthorize access, Optimization (paging, swapping, caching)
*Storage (File) Management*: Abstract storage into files / directories
- Task: Create / delete / open files. Enforce permissions (rwx). Backup, fragmentation management.
*Device Management* I/O Subsystem *&* Mass-Storage Management
- _IO Subsystem_: Buffering (temporary storage for data in transit), Caching (store frequently used data in RAM), Spooling (store data for later use), Device Drivers (interface between OS and device)
- _Mass-Storage_: Free-space management, Storage allocaiton, Disk scheduling (minimize head movements)
*User Command Interface*: CLI, GUI. Translate user input $->$ syscall

#v(1em)
#head([Interrupts & System Calls])

#subhead([Interrupt Processing])\
*Basic Idea*: Major event $->$ Signal to the CPU $->$ Seize the CPU
*Priority*: 
- _Maskable Interrupt_: low priority. Ignored / handled later
- _Non-Maskable Interrupt_ (*NMI*): high priority. Handled immediately
  - Higher priority NMI, can interrupt lower priority NMI
  - Otherwise, low priority NMI waits for prev NMI to finish

*Procedure*: 
+ Non-maskable interrupt occurs
+ Save current CPU state (PC, Register) to stack
+ Look up an _interrupt vector table_ (*IVT*) to find appropriate ISR
+ [User → Kernel Mode]: Jump to _Interrupt Service Routine_ (*ISR*)
+ *[In Kernel Mode]:* Execute ISR   
+ Restore CPU state from stack
+ Re-enable interrupts

*Interrupt Types*

#underline[[1] _Program Interrupt_]
- Windows: `Ctrol-Alt-Del` $->$ Secure Attention Sequence (SAS) $->$ Hardware Interrupt. Major, unique system-level interrupt.
- Unix: `Ctrol-C` $->$ Software Interupt (User Trap). General purpose software termination
- Hardware Interrupts are reliable. BUT, impossible to catch
- Software Interrupts let program handle interrupt (cleanup resource)
#figure(image("dual-mode-op.png"), caption: "Dual Mode Operation: distinguished by *mode bit*")

#underline[[2] _I/O Interrupt_]
#figure(image("io-interrupt.png"), caption: "CPU interrupt needed 1 tick after transfer complete")
Must know calculations: 
- Sync: Total time = sum (e.g., 128ms for 20+5+3+100).
- Async: Total time ≈ longest task (100ms), more interrupts.

Device-Status Table: 
- Managed by kernel. 
- Each entry: contains type, address, state
- State updated after an interrupt (not function, idle, busy)

#underline[[3] _Timer Interrupt_]
+ Interrupt occur after specific time period
+ Kills process / take back resource (e.g., unlock file)
+ In Unix: _clock routine_. Trigger by hardware per $1/60$ sec

#head([System Calls])\
*Goal*: Abstract away hardware details. Provide interface for user.

#subhead([Procedure])
+ User process executes syscall
+ Look up _System Call Table_
+ CPU switch to kernel mode and execute syscall
+ Return status & results. 

#subhead([Types])
#image("syscall-table.png")

#subhead([System Programs])
+ *File Management*
  - Functions: Create, delete, copy, rename, print, list, and manipulate files/directories.
  - Example: `ls` command in UNIX lists directory contents.
+ *Status Information*
  - Functions: Retrieve system data (e.g., date, time, memory, disk space, user count).
  - Complex versions: Performance monitoring, logging, debugging.
  - Output: Terminal, GUI, or files.
  - Example: `top` command in Linux displays system resource usage.
+ *File Modification*
  - Tools: text editors, search command, text transformation utilities.
  - Example: `vim` for editing files, `grep` for searching file contents.
+ *Programming-Language Support*
  - Tools: Compilers, assemblers, debuggers, interpreters.
  - Example: GCC for C/C++, Python interpreter.
+ *Program Loading and Execution*
  - Tools: Loaders (absolute, relocatable), linkage editors, overlay loaders, debugging systems.
  - Example: `gdb` for debugging C programs.
+ *Communications*
  - Functions: Facilitate virtual connections between processes, users, and systems.
  - Examples: Email, remote login (`ssh`), file transfer (`scp`).
+ *Background Services (Daemons)*
  - Functions: Long-running processes for essential tasks (e.g., network connections, process scheduling, error monitoring).
  - Examples: Network daemons, print servers.
  - Note: Daemons often run at boot time and continue until the system shuts down.

#subhead([Parameter Passing])
#grid(columns: 2, gutter: 0em,
  [#image("memblock-pass-by-ref.png")],
  [
    + *Registers*: fast, but limit space.
    + *Memory Block/Table*: 
      - Put param in mem block
      - Put address of param mem blocks in registers
    + *Stack*: for variable-len data.

  ]
)

#subhead([OS Types & Structure])
#image("os-types-table.png")
#place(bottom, image("os-types.png", width: 125%), dx: -20%)
*Real-time*: Processes have completion deadlines must be met\
*Distributed*: Operate upon collection of computer across network
#image("os-types-table1.png")

#head([I Hate Shell Script])\
*`ls` usage*
- `ls -l`: list in long format (detailed, verbose)
- `ls -a`: list all (include hidden files)
- `ls -R`: list subdirectories recursively
- `ls [hello]`: match single-character file `h`, `e`, `l`, *or* `o`
- `ls [hello]*.c`: files starting with `h`, `e`, `l`, `o` and ending with `.c`
- `ls hello[1-3]*`: file starting with `hello` followed by `1`, `2`, or `3`
*Pipe*: `(ls; cal) | wc` - (ls; cal) creates a subshell; within this subshell, `ls` and `cal` run in two subprocesses (so 3 processes in total on the left hand side). Then, output piped to `wc` which runs in another 

*Unfamiliar Commands*:
- `echo something\`: `\` allow continue typing command on new line
- `echo $$`, `$$` evaluates to PID of current shell 
- `wc`: outputs line count, word count (by space separate), *bytes* count, file name (if called with file name as parameter, eg. `wc test.txt` )
- `source`: execute a script or batch file
- `history`: list recent cmds; `!n` run command n; `!!` run last command
- `| more`: pipe output to `more` to display one screen at a time
- `export`: make variable available to subshells, but not parent shell
- ``` `cmd` ``` or `$(cmd)`: command substitution. ``` ` ``` has undefined behavior
- Exponential in shell is `**`, not `^`
- `ps -ef | more` will display detailed information about all processes
*Special Parameters*
- *`$*`*: All positional parameters as a single string
- *`$@`*: All positional parameters as separate strings
- *`$1, $2, ...`*: Positional params (arguments to script), start from 1
- *`$#`*: Number of positional parameters (arguments)
- *`$?`*: Exit status of the most recently executed command
- *`$$`*: Process ID of the shell
- *`$0`*: Name of the shell or shell script
- *`${!#}`*: evaluates to the last argument
*Exist Status*: 0 for success, non-zero for failure, -1=255, 256=0\
*Access Privilages*: [dir or file] [user] [group] [other] `-rwx r-- r--`
- `chmod g+r f`: give group read access to file
- `chmod o-r f`: remove other's read access to file
- `chmod u+x f`: give user execute access to file
- `chmod -R g+rw folder`: recursively give group read/write access to folder and all its content within.
- `chmod 777 f`: give all access to file
- Each digit represents user, group, other. E.g., $6=(110)_2="rw-"$
- *First character (before `rwx`, eg `drwx-rwx-rwx`)*: 
  #grid(columns: 2, gutter: 1em, [
    - `p`: FIFO (named pipe) 
    - `-`: Regular file 
    - `d`: Directory
    - `l`: Symbolic link
  ], [
    - `s`: Socket
    - `c`: Character device
    - `b`: Block device
  ])

`$PATH` separated by `:`  || To add current dir: `PATH="$PATH:."`

#def(
[
*Regex*: 
- `^` means beginning of line; 
- `$` means end of line; 
- `.` means any character;
- `[abc]` any one of a, b, c; 
- `[^abc]` anything except a, b, c; 
- use `\` for escape.
`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
],
[
*`grep` Commands*:
`grep poly *.c` list all C files containing `poly`;\
`grep -i poly *.c` same above, but case insensitive;\
`grep -l -i '^poly' *.c`  print only *names* of file with lines beginning with poly\
`grep -h -i '^poly' *.c` print only lines beginning with poly\


],
g: 1em
)
- Interpreted: PHP, Ruby, Perl, Bash, Lua, Tcl, Basic, HTML, JS, Py
- Compiled: COBOL, Fortran, Pascal, Ada, Lisp
*Scripting Language*: 
- Advantages: fast to write, small size, partial execution, directly access OS service and execute commands, directly process output from other programs;
- Disadvantage: slow execution, weak data structure / typing, odd syntax, error-prone, occasionally buggy

*Random Concepts*
- Run commands together: `cmd1; cmd2; cmd3`
- Run cmd if previous success: `cmd1 && cmd2`
- Global variables by convention UPPERCASE, local lowercase
- For local variables use ```bash local var=value```
- *DO NOT* add space before and after `=`
- *DO NOT* create a script called `test`, interfer with built-in `test` cmd
- *`let`* keyword
  - `let x=a+b` is equivalent to `x=$((a+b))`
  - but `x=a+b`, x is the literal string `a+b`
- *changes to the list* inside the loop will _*not*_ affect the loop count

*Special Option Variable*
- *`history`*: enable command history, default = on
- *`noclobber`*: prevent overwrite file when I/O redirection, default = off
- *`ignoreeof`*: stop shell from exiting when EOF is sent, default = off
- *`allexport`*: _automatically_ export all modified variables, default = off
- To turn on/off, use `set -o/+o variable`: 
  - `set -o noclobber` (turn on noclobber feature)
  -  `set +o noclobber` (turn off noclobber feature)


*Built-in Variables*
- *`HOME`*: Full pathname of home dir; default destination for plain `cd`
- *`PATH`*: list of directories for executables and scripts, `:` separated
- *`PS1`*: Primary prompt string; default '\s-\v\$ ' (system-version then `$`).
- *`PS2`*: Secondary prompt string for line continuation; default '> '.
- *`PWD`*: Current working directory (updated by `cd`).
- *`HOSTNAME`*: Current machine name.
- *`UID`*: Internal Unix user ID. *`PPID`*: Parent process ID.
- *`HISTSIZE`*: Max history lines; default 500.

*Magic Numbers*
- `#!` is _magic number_. Look at the path following `#!` and start that program as an interpreter.
- _Big Endian_: Stores the most significant byte first (same order as human-readable representation).
- _Little Endian_: Stores the least significant byte first (reverse order relative to human-readable
representation).
- `od -c -x file | more` to see magic number of file
  - `od` means octal dump, `-c` means char, 
  - `-x` means hex in little-endian, `-t x1` means hex in big-endian
  #place(auto, float: true, image("octal-dump-output.png"), dx: -3%)

*Array*
- `declare -a arr`: create an empty array
- `arr=(1 2 3 4 "str")`: create array
- `${arr[2]}`: access element (index start from 0)
- `${arr[@]}`: access all elements, `${#arr[@]}`: length of array
- `${arr[@]:1:2}`: slice array from index 1 to 2
- *`@` vs. `*`*: completely the same unquoted. Different when quoted.
  - `"${arr[@]}"`: five strings || `for i in "${arr[@]}"`, output 5 lines.
  - `"${arr[*]}"`: one string || `for i in "${arr[*]}"`, output 1 line.
- *single quote `'`*
  - no variable expansion: `echo '$HOME'` prints `$HOME`
  - no command substitution: `echo '$(ls)'` prints `$(ls)`
  - no escape sequence: `echo '\n'` prints `\n`,
  - no special character interpretation: `echo '!'` prints `!`
- *double quote `"`*
  - Variable expansion: `echo "$HOME"` prints `/home/user`
  - Command substitution: `echo "$(ls)"` prints output of directory
  - Arithmetic expansion: `echo "$((2+3))"` prints `5`
  - Escape sequence: `echo "\n"` prints newline
  - *no quotes*: 
    - `${arr[@]}` or `${arr[*]}` behave the same 
    - `${#arr[@]}` or `${#arr[*]}` behave the same
    - Same as `expand(' '.join().split())`
      - `expand(arr)` is variable expansion on each element of `arr`
  - *with quotes + `[@]`*: `"${arr[@]}"`: preserve list as is
  - *with quotes + `[*]`:* `"${arr[*]}"`: equivalent to `' '.join(arr)`
- `declare -a arr`: declare array || `declare -p arr`: print array

*Conditionals*
- `if [ condition ]; then ... fi`.
- `[]`, POSIX-compliant. Variables must be quoted to handle space.
- `[[]]`, bash-specific. 
  - Allows pattern matching (`[[ $var == pattern* ]]`), 
  - regex (`[[ $var =~ ^[0-9]+$ ]]`) 
- Logic operators (`[]` and `[[]]` both supports):
  - `!`, `-a` (&&), `-o` (||). -eq(==), -ne, -gt, -ge, -lt, -le

*Case-Statement*
#def([
```bash
case $code in
  comp1*) echo "level 1" ;;
  comp2*) echo "level 2" ;;
  comp[34]*) echo "senior" ;;
  comp*) echo "invalid code" ;;
  *) echo "bruh what?" ;;
esac
```
],[
  - Each pattern ends with `)`
  - Pattern: constant or wildcards
  - Case sensitive
  - `;;` no fall through. 
  - `;;&` fall through
], g: 0pt)

*For-loops*: `for var in "${list[@]}"; do ... done`\
*While-loops*: `while [ condition ]; do ... done`\
*Until-loops*: `until [ condition ]; do ... done`

*File Testing*: `-e` exists, `-d` is dir, `-f` is plain file, `-r` readable, `-w` writable, `-x` executable, `-z` empty

*Numeric Calculation*: `result=$((expression))` 

*Input/Output*
- `read var`: read input from user, store in var
- `read -p "Prompt: " var`: print prompt, then read input
- `read -a arr`: read elements into an array "`arr`"
- Redirection *(Sequential!)*
  - `<` and `>`: redirect input and output. 
  - `script < input.txt > output.txt`
  - `>>`: append to file, `&>>` same as `>>` but also *append error*
- Pipes *(Concurrent!)*
  - `p1 | p2` $equiv$ `p1 > tmp; p2 < tmp`
  - `p1 | p2 | p3` $equiv$ `p1 > tmp1; p2 < tmp1 > tmp2; p3 < tmp2`
  - `p1 < in | p2 > out` $equiv$ `p1 < in > tmp; p2 < tmp > out`

#image("linux-summary.png")

#head([Process Management])
#place(auto, float: true,
grid(columns:2, [#image("lifecycle-small1.png", width: 100%)], [#image("lifecycle-small2.png", width: 130%)], gutter: 0pt),
dx: -4%, dy: -1%
)
#place(auto, float: true, image("process-lifecycle.png", width: 120%), dx: -4%, dy: -0%)
- *LTS* aka _Job Scheduler_: which process to enter _ready queue_
  - Controls the degree of multiprogramming
  - No long-term scheduler in Unix and Windows
- *STS* aka _CPU Scheduler_: which process to allocate CPU next
- *MTS* when too many processes compete CPU, remove some from ready queue; When few process in CPU, add them back 
  - _MTS_ can cause _*Thrashing*_: when CPU spends more time swapping than executing
- Cooperative Process: share variable, mem, code, resource (CPU, I/O)
  - need synchronization
- Independent Process: no shared resource

#subhead([Cooperative Process])
- *Producer-consumer* Streaming, Web Server
- *Reader-writer* Example: Banking Systems
- *Dinning Philosopher*
  - Cooperating process that need to share limited resources

*Process Control Block (PCB)*
- Process State: New, Ready, Running, Waiting, Terminated
- Program Counter: address of next instruction
- CPU Registers: accumulators, index registers, stack pointers
- CPU Scheduling Information: priority, scheduling queue
- Memory Management Information: limit of memory boundary
- Accounting Information: CPU used, process ID
- I/O Status Information: list of open files, devices

*`exec()`*: replace current process with new process: 
  - *Same*: PID, file descriptor, cwd, user / group ID, resource limit 
  - *Overwritten*: mem, stack, data, heap, program counter, registers
#place(top, float: true, image("exec-family.png", width: 115%), dx: -8%)
*`fork()`*: create new process (child) from existing process (parent)
  - *Parent*: return child PID, child PID > 0, error return -1
  - *Child*: return 0. Call `getppid()` to get parent PID
*`wait()`*: parent blocked until *one of* the childs terminates

#head([Interprocess Communication (IPC)])

#subhead([General Concepts])\
*Indirect Communication*:
- Messages are directed and received from _mailboxes_ aka _ports_
*Direct Communication*:
- Process must name each other directly 
  - `send(P, msg)` `receive(Q, msg)`
- Links established automatically by OS. Each link associate with *exactly one pair* of processes. 
*Blocking* (_Synchronous_):
  - Sender blocked until message received by receiver
  - Receiver blocked until message is available from the sender
*Non-blocking* (_Asynchronous_):
  - Sender sends message and always continue
  - Receiver receives valid message or `null` if message not available
*Buffering* (_Message Queues_) -- sender sent, recv haven't read yet
 - Zero Capacity: No msg buffered. Sender must wait for receiver, v.v.
 - Bounded Capacity *(common)*: Fixed size queue. Sender wait if full.
 - Unbounded Capacity: Infinite size queue. Sender never wait.
*Process*: Top-level execution container. Separate Memory Space\
*Threads*: Runs _inside_ a process. Share Memory Space

#subhead([Unix-specific IPC])\
*Shared Memory Mechanism* (faster than message passing): 
- *Method 1*: Special Syscalls to create / remove / write / read from shared kernel memory space. || *Method 2*: Use `Pthreads` library, whose threads naturally share memory space due to same process

*Message Passing Mechanism*: _pipes_ or _sockets_
- `write` one way. `talk` two way.
- Pipes: _Unnamed pipes_ OR _Named pipes_
  - Unidirectional, FIFO, unstructured data stream
  - Fixed size Buffer allocated in kernel.
  - *_Unnamed Pipes_*: 
    - Create by syscall `pipe()`. Kernel buffer allocated.
    - Two file descriptors (fd) as a 2-element array
      - `fd[0]` read end, `fd[1]` write end
      - Process will read from `fd[0]` and write to `fd[1]`
      - e.g., ```cpp read(fd[0], buf, count)```, ```cpp write(fd[1], buf, count)```
    - *Usage*: 
      + First create `fd[]` then `fork()`. 
      + Now, both child and parent have access to `fd[]`
      + `close(fd[0-or-1])` in child and parent (removes their access, does not actually destroy the file descriptors)
  - *_Named Pipes_*
    - _named pipe_ is an actual file created by `mkfifo()`
      - `mkfifo()` return 0 on success, -1 on error
    - After creation, any process knowing the name can `open()` it
    - They can then communicate via `read()` and `write()`
    - pipe won't delete when processes end. Must use `unlink(pipe)` at the end of process to remove it (good practice)
  - *Choices*
    - _Unnamed pipes_ when only parent-child communication, safe
    - _Named pipes_ when unrelated processes or across sessions
    
*Syscalls*\
_*Below uses ```cpp #include<unistd.h>```*_
- *```cpp ssize_t write(int fd, const void *buf, size_t count);```*
  - `fd`: file descriptor, `buf`: data to be written, `count`: number of bytes to write. If `count` > content in `buf`, write random shit at end.
- *```cpp ssize_t read(int fd, const void *buf, size_t count);```* 
  - `fd`: file descriptor, `buf`: where data saved, `count`: \# of bytes to read

_*Below uses ```cpp #include<sys/types.h> #include<sys/stat.h>```*_
- *```cpp int mkfifo(const char *pathname, mode_t mode)```*
  - `mode = 0777`, means `prwxrwxrwx`
- *```cpp int open(const char *pathname, int flags)```*
  - Also needs `#include <fcntl.h>`
  - flag for reading `O_RDONLY`, for writing `O_WRONGLY`, for r+w `O_RDWR`
- *```cpp int open(const char *pathname, int flags, mode_t mode)```*
  - `mode` is the mode to open the file -- reading or writing
  - Also needs `#include <fcntl.h>`


#subhead([Communication Topology])

*Cables* 
 - *Star Topology*: $n-1$ cables. Bottleneck at parent. Few pipes. 2 hops needed at max.
 - *Ring Topology*: $n + 1$ cables. Many hops. Few pipes.  
 - *Linear Topology*: $n + 1$ cables. Many hops. Few pipes
 - *Fully Connected*(fc): $(n(n - 1))/2$ cables. Many pipes. Single hop
 - *Tree Topology*: Few pipes. Many hops (log?)
 - *Mesh Topology*: $(n(n-1))/2$ cables. Follow some structure (e.g., grid). Less pipes then fc. Well established non-neighbor routing. Flexible. 
 - *Hypercube Topology*: Fewer pipes than fc. Well established non-neighbor routing. Flexible.

*Number of pipes*\
• *Star*: 2(n-1) pipes. Bottleneck at parent node. Single-point-of-failure.
• *Ring*: 2n pipes. No particular bottleneck. Distributed communication.
• *Linear*: 2(n-1) pipes. Node failure separates the network.
• *Fully connected*: n(n-1) pipes. No bottleneck. Each node has 2(n-1) connections.
• *Tree*: 2(n-1) pipes. Bottleneck spread throughout hierarchy.
• *Mesh*: Between 2(n-1) and n(n-1) pipes. Each node connects to ~4 neighbors.
• *Hypercube*: Between 2(n-1) and n(n-1) pipes. Each node connected to $log_2(n)$ neighbors

#head([CPU Scheduling])

#subhead([Basic Considerations])
- *CPU utilization*: % of CPU time used for real work.
- *Throughput*: \# of processes completing execution per time unit.
- *Turnaround time*: Time to finish process since arrival. 
- *Waiting time*: Time a process spends waiting in ready queue
- *Response time*: Time from request submission until first response.
- *Completion Time*: The time when completed. 

#subhead([Scheduling Algorithms])
- *Preemptive*: Priority Scheduling, Round Robin (RR), Shortest Remaining Time (SRT)
- *Non-preemptive*: First Come First Serve (FCFS), Shortest Job First (SJF), Multi Level Queues

*Non-preemptive*:
- *First Come First Serve (FCFS)*: First process in, first process out.
- *Shortest Job First (SJF)*: Shortest job first.
- *Multi Level Queue*: Ready queue devided into multiple queues.
  - Each queue has own scheudling algoirhtm
  - Scheduling *between queues*. 
    - Fixed Priority Scheduling: each queue has own priority
    - Time Slicing: each queue gets a time slice (like RR)
  #image("multi-level-queue.png", width: 60%)

*Preemptive*: 
- *Shortest Remaining Time (SRT)*: Preemptive version of SJF.
- *Round Robin (RR)*: 
  - Predefined _time quantum_ (e.g., $q=10"ms"$).
- *Priority Scheduling*: 
  - _Priority Number_ assigned to each process. Higher priority first.
  - SJF is priority scheduling where priority is the time to completion.
  - FCFS is priority scheduling where priority is the time of arrival.

*Strengths & Weaknesses*
 - SJF / SRT: Optimal for minimizing waiting time, good for batch jobs.
  - SJF proven to have lower TAT, *optmal* for non-preemptive
  - SRT proven to have lower TAT, *optimal* for preemptive
 - RR: fair to all process, good for interactive jobs.
 - FCFS: simple, no starvation since all will eventually execute. But _Convoy Effect_, where short process stuck behind long process.

#subhead([Common Scheduling Issues & Solutions])

*Issue 1: Starvation*
- *Problem:* Low-priority processes may never execute due to continuous arrival of high-priority processes
- *Solution:* Implement _aging_ mechanism - processes gradually increase in priority the longer they wait

*Issue 2: High Overhead*
- *Problem:* Complex scheduling algorithms (e.g., multilevel feedback queues) create significant system overhead
- *Solution:* Optimize queue management with efficient data structures and algorithms

*Issue 3: Priority Inversion*
- *Problem:* Higher priority process waits while lower priority process holds a needed resource
- *Solution:* Implement _priority inheritance_ - lower priority process temporarily inherits the higher priority

*Issue 4: Unpredictability*
- *Problem:* Execution order can be unpredictable, problematic for real-time systems requiring timing guarantees
- *Solution:* Use deterministic scheduling algorithms or provide explicit timing guarantees for real-time requirements

*Issue 5: Parameter Tuning*
- *Problem:* Selecting optimal values (time quantum, priority levels) significantly impacts performance
- *Solution:* Systematic testing and performance analysis to determine optimal parameters

#head([Memory Management])
- *Logical Address*: virtual address, generated by CPU
- *Physical Address*: absolute address. address sent to memory units.

*Binding*: logical $->$ physical addr translation
- _*Compile-time*_: Phys addr hardcoded. Recompile if program moved.
- _*Load-time*_: _Relocatable code_ (relative addr). Phys addr calculated when loaded to RAM; address fixed afterwards.
- _*Execution-time (Run-time)*_: Bind delayed till run. Process can move during execution. Logical addr used; MMU calc phys addr dynamically. Needs HW support (MMU).

*Memory Management Unit (MMU)*
 - _Relocation Register_ (RR): added to every logical addr to get base.
 - _Limit Register_ (LR): offset limit from the base physical memory
*Contiguous Mem. Allocation*: Entire process in single block of mem
- Multiple fixed partition: multiprogramming w/ fixed num. of tasks
- Mult. variable partition: multprogramming w/ varying num. of tasks
  - Process arrives $->$ finds *hole* large enough to fit it
  - First-fit: find first hole that's big enough. 
  - Best-fit: find smallest hole big enough (little leftovers, not useful)
  - Worst-fit: find largest hole (large but useful leftovers)
  - *First-fit* and *Best-fit* are _normally_ better
  - _*External Fragmentation*_: space exist, but not contiguous (fixed with _compaction_ - stop all process and group memory together)
  - _*Internal Fragmentation*_: waste within partition (hole slightly larger than process, and managing this overhead is not worth it)
*Non-Cont. Mem. Alloc*: multiple blocks (_paging_ or _segmentation_)
- *_Paging_*: physical mem divided into fixed-size *frames*. Logical mem divided into same fixed-size *pages*. Page table maps page $->$ frames
  - logical addr. = page number + offset 
  - _Page number_: index of page table. _Offset_: offset within page.
  - #underline[Paging suffers Internal Fragmentation]
  - #underline[Paging allows *sharing* a program across multiple processes]
  - *Implementation*: 
    - _page-table base register_ (PTBR): points to page table of a process in the memory. *Fast context switch*: just change PTBR.
    - But needs #underline[two mem. accesses]: 1. get page table, 2. get the data.
    - *Solution*: _Translation Look-aside Buffer_ (TLB) - store recently translated logical page numbers.
      - cache access time $c$, memory access time $m$, TLB hit ratio $h$
      - Translation time = $h times c + (1-h) times (c+m)$
      - Effective access time = $c + (2-h) times m$
  - *Hierarchical Paging* (otherwise, page table too large): 
    - Example, 32bit machine - 20bit for page number, 12bit for offset.
    - Page the page table: 10 bit page number, 10 bit offset
  - *Memory Protection*: 
    - _Valid-Invalid bit_: additional bit to each page table entry
    - _Page-table length register (PTLR)_: check validity of logical addr.
- *_Segmentation_*: logical memory space is different-sized segments. Each segment table entry contains _base_, _limit_, _valid bit_ 
  - Segment-table base register(STBR): point to segment table base in memory. _Segment-table length register (STLR)_: number of segment
  - #underline[Segmentation suffers External Fragmentation]
  - #underline[Segmentation also allows sharing. Put same entry in each table]

#head([Virtual Memory])
*Techniques*
- _*Overlay*_: break program into stages, and load them sequentially
  - swapping but on different *stages* instead of parts of program
  - Memory has a common area for common modules
  - When program $>$ physical memory (eg embedded systems)
- _*Vritual memory*_: store process on disk, load on demand.
  - Uses *memory map* (basically page table but for virtual mem)
  - More genereal puposed
*Checking memory usage*
- `a.out &` runs `a.out` in background. #underline[*allow multiple instance*]
- Then `ps -l` lists current running process in this terminal. 
- Columns: `VSZ` total size. `RSS` mem usage (`VSZ` $>=$ `RSS`)
*Methods*:
- _Demand paging_: load page only when it's needed.
  - Need for a page indicated by PC or MAR. 
  - Via lazy process _*swapper*_. Swapper that swaps pages = _*pager*_.
  - Valid-invalid bit: indicate whether page is in memory
  - *Procedure*: 1. check valid-invalid bit. 2. if invalid, trigger a _trap_ called _page fault_. 3. Run _page fault handler_, that ask _pager_ to swap
    - Note: valid-invalid bit set to invalid for all entries initially.
    - #underline[ Page fault handling]: 1. get empty frame from free frame list. 2. load page from disk. 3. update page table to point to new frame. 4. set valid-invalid bit to valid. 5. Restart page fault instruction.
  - *Performance*: 
    #def([
      - page fault rate = $0 <= f <= 1$, 
      - memory access time = $m$, 
      - page fault service time = $s$.
      - Effective Access Time (EAT): expected memory access time
    ], 
    [ 
    page fault service time = page fault overhead + time to swap page in + restart overhead
    $ "EAT" = (1-f) times m + f times s $
    ])
- _Anticipatory paging_: predict what page will be needed next.

#subhead([Page Replacement])
- _Belady's Anomaly_: page fault rate $arrow.t$ if number of page frames $arrow.t$
- _Reference String_: page number sequence
- *FIFO*: Always replace oldest page. Suffer from _Belady's Anomaly_. 
- *Optimal*: Replace page that will be used furthest in the future.
- *LRU*: Replace page that has not been used for the longest time.
- *Allocation*: 
  - _Local page replacement_: process only replaces its own frames
  - _Global page replacement_: process can replace other's frames
  - _Fixed allocation_ (#underline[*only _local page replacement_ allowed*]): 
    - Equal allocation: each process gets same number of frames ($F/n$)
    - Proportional allocation: each get frames proportion to its size
  - _Variable allocation_: number of frames vary over time. Give processes with too many page faults more frames.
*Thrashing*: when number of available frames $<$ total size of active pages of the processes in the ready queue. 
  - Solution: *reduce* the degree of multiprogramming.
  - _Working-Set_: total number of unique pages reference in a period $Delta$
  - When working-set size $>$ number of frames, thrashing occurs
  - _*Page Fault Frequency (PFF) algorithm*_: define acceptable range. If rate too low, takeaway some frames. If rate too high, add frames.

#head([Deadlock])
- *Deadlock*: set of blocked resources. cannot proceed. *Conditions: *
  - _Mutual Exclusion_: only one process can use a resource at a time
  - _Hold and wait_: process holds $>=$ 1 resource, waits another resource
  - _No preemption_: resource only released voluntarily by process
  - _Circular wait_: hold-and-wait processes forms a cycle.
  #place(right, image("resource-allocation-graph.png", width: 40%))
- *Livelock*: can be resolved by chance.
#subhead([Resource Allocation Graph (RAG)])
 - $P = {P_1, P_2, ... P_n}$, set of all processes
 - $R = {R_1, R_2, ... R_m}$, set of all resources
 - *Request Edge*: $P_i$ requests $R_j$ ($P_i -> R_j$)
 - *Assignment Edge*: $R_j -> P_i$
 - If a req can fulfill, #underline[_*request edge becomes\ assignment edge*_]
*Check for deadlock (RULES)*: 
 - If no cycle, #underline[_definitely no deadlock_]
 - If cycle, only one instance per resource type, #underline[_deadlock_]
 - If cycle, multiple instances per resource type, #underline[_might deadlock_]

#subhead([Deadlock Handling])
- _Ostrich approach (Deadlock ignorance)_: ignore the deadlock
- _Deadlock avoidance (Banker's algorithm)_: don't enter deadlock state
- _Deadlock prevention_: ensure never enter deadlock state
- _Deadlock detection_: detect deadlock after happen; recover from it

*Deadlock Prevention*: 
 - Ensure when process request resource, it is not holding other resources (#underline[in-degree/out-degree cannot be non-zero at the same time])
 - Make preemption possible (e.g., by using time quantum)
 - Prevent circular wait: 
  - Order all resource types with a pre-defined number
  - $P_i$ can only request $R_j$ if $j$ larger than the number of any resource currently held by $P_i$. Else $P_i$ have to release larger resource first
  - (#underline[All edge point from small number R to large. No reverse edges])

*Deadlock Avoidance* (_Banker's algorithm_): 
- _Safe State_: system is safe if exist certain resource allocation order that can avoid deadlock. This order is called _*Safe Sequence*_
- ${P_1, P_2, dots, P_n}$ is safe sequence if: for each $P_i$, its _need_ can be satisfied by current available resource + resources held by all $P_(j<i)$

*Deadlock Detection (_Wait-for Graph_)*
- Detect deadlock by _deadlock detection algorithm_
- Execute _deadlock recovery scheme_ to recover from deadlock
  - _Process termination_: kill all deadlocked processes / kill one victim process at a time until deadlock eliminated (consider: priority, progress, etc.) Might _*starvation*_: some process always victim
  - _Process rollback_: return to safe state and retry




#subhead([Banker's Algorithm])
#def(
  [
    - $"Avail" [m]$\
    - $"Alloc"_i [m]$\
    - $"Max"_i [m]$\
    - $"Need"_i [m]$\
  ],
  [
    \# of resources $m$ currently available\
    \# of resources $m$ currently allocated to process $i$\
    \# of resources $m$ that process $i$ will at most need\
    $"Max"_i [m] - "Alloc"_i [m]$ (how much more $i$ needs)\
  ],
  g: 1em
)
#place(right,
pseudocode-list(booktabs: true)[
  + $"Work" <- "Avail"$ (array of length $m$)
  + $"Finish" <- ["False", ...]$ (array of length $n$)
  + *while* not done *do*: 
    + find $i$ s.t. $not "Finish"[i] "and" "Need"_i <= "Work"$
    + if no such $i$ exists, *break*
    + $"Work" <- "Work" + "Alloc"_i$
    + $"Finish"[i] <- "True"$
  + *if* $"Finish"[i] = "False" forall i$ *then* return _Safe_
  + *else* return _Unsafe_
]
)
*Intuition*: 

+ Simulate allocate\ resource to a $P_k$
+ Assume $P_k$ run in\ background (async)
+ *If* (synchronized)\ execution sequence\ exists then *safe*\ Else *unsafe*

#v(2em)
*Find all safe sequence*: requires drawing search (DAG) graph.

#subhead([Wait-for Graph])\
RAG except simplify *FROM* $P_i -> R_j$ and $R_j -> P_i$ *TO* $P_i -> P_j$
Periodically check for cycles in the _Wait-for Graph_

#head([Synchronization])

*Critical Section (CS) Problem*: 
- _Critical Section_: code segment where shared data is accessed
- Solution: [_Entry Section_] + [_Critical Section_] + [_Exit Section_]
- *Rules*: 
  + _Mutual Exclusion_: only one process in their CS at a time
  + _Progress_: When no one in CS, and one process wants to enter CS, decision must be made within finite time. (*no hanging request!*)
  + _Bounded Waiting_: Between request and grant for A, limit the number of times B enters CS. (*fairness / no starvation!*)
- *Solutions*: 
  - *Lock Variable*: shared boolean 
    ```cpp
    while (lock == true) wait();
    lock = true; Enter_CS(); lock = false;
    ```
    *Problem*:
    + Process $P_0$ checks lock; it's false.
    + Context switch *before* $P_0$ can set lock = true.
    + Process $P_1$ checks lock; it's still false.
    + $P_1$ sets lock = true and enters its critical section.
    + Context switch back to $P_0$ (already passed while). Sets lock = true (even though it's already true) and also enters its CS.
    + Now both $P_0$ and $P_1$ are in their CS simultaneously!!!

  - *Test-and-Set (TSL) Lock*: use special hardware instruction, TSL. This instruction is atomic. #underline[It is a single, *indivisible* operation]
    ```cpp
    while(TestAndSet(&lock) == true) wait();
    Enter_CS(); lock = false;
    ```
  
  - *Turn Variable*: two processes $P_0$ and $P_1$. A shared integer variable *turn* to indicate #underline[whose turn it is to enter the critical section]
    ```cpp
    while(turn != self) wait();
    Enter_CS(); turn = other;
    ```
  #table(
    columns: 4,
    inset: (x: 4.5pt, y: 1.5pt),
    stroke: (x: none, y: none),
    table.hline(stroke: 0.5pt + black),
    [*Solution*], [_Mutual Excl._], [_Progress_], [_Bounded Wait_],
    table.hline(stroke: 0.5pt + black),
    [*Lock Variable*], [Fails], [Fails], [Fails],
    [*Test and Set*], [Satisfied], [Satisfied], [Fails],
    [*Turn Variable*], [Satisfied], [Fails], [Satisfied],
    table.hline(stroke: 0.5pt + black),
  ) 

  #subhead([Peterson's Algorithm])

  #pseudocode-list(booktabs: true, title: [*Peterson's Algorithm* ($"flag"$ helps satisfy _Progress_)])[
    + $"turn" <- 0$ \/\/ indicate whose turn to enter CS
    + $"flag"[0] <- "flag"[1] <- "false"$ \/\/ whether process ready to enter CS
    + *while* true *do*:
      + $"flag"[i] <- "true"$
      + $"turn" <- j$ \/\/ Use $j$ to indicate other processes ($j != i$)
      + *while* $"flag"[j] = "true"$ and $"turn" = "j"$ *do* 
        wait() *end*
      + #text(fill: red)[[*Critical Section*]]
      + $"flag"[i] <- "false"$
      + *Remainder Section* $dots$
  ]
  *Pros*: Builds on top of *Turn Variable* and satisfies _Progress_\
  *Cons*: while-loop waste CPU time. The thing it's waiting starved.
  
  #subhead([Semaphores]) - Ultimate solution for critical section problem

  #grid(columns: 2, gutter: 0em,
  ```cpp
  while (true) {
    P(S);
    Enter_CS();
    V(S);
    Remainder_Section();
  }
  ```,
  [
    *S* is a semaphore, shared by all processes\
    *P(S)*: decrease $S$ by 1, if negative move current process to waiting queue\
    *V(S)*: increase $S$ by 1, if exist processes in waiting queue, wake up one of them
  ]
  )

  *Intuition*: 
  - *S* is a counter of available resources
  - *P(S)* check if any resource available. If true, take one (*S* -= 1). Otherwise (*S* $ <= 0$), this process sent to waiting queue
  - *V(S)*: called when process release a resource (*S* += 1). If there is any process in waiting queue, wake up one of them.
  - Both *P(S)* and *V(S)* are #underline[atomic operations]. Can't context switch.
    - *P(S)*, aka: `wait()`, `down()`, `acquire()`
    - *V(S)*, aka: `signal()`, `up()`, `release()`

  *Other Concepts*
  - _Counting Semaphore_: $[-oo, +oo]$, though typically non-negative. 
    - For resource that has multiple instances
  - _Binary Semaphore_ (*Mutex*): $[0, 1]$, Init. to 1, alter between 0 and 1.
    - For resource that has only one instance.
  
  #subhead([Producer-Consumer Solution (using Semaphores)])

  #grid(columns: 3, gutter: 1em,
  ```cpp
  // producer
  while (true) 
  {
    down(empty);
    down(mutex);
    produce();
    up(mutex);
    up(full);
  }
  ```,
  ```cpp
  // consumer
  while (true) 
  {
    down(full);
    down(mutex);
    consume();
    up(mutex);
    up(empty);
  }
  ```,
  [
    *`empty`:* \# of empty buffer slots\
    *`full`:* \# of occupied slots\
    *`mutex`:* ensure only one party access the buffer at a time\

    If `wait(mutex)` placed before `wait(empty)`, producer can lock buffer and do nothing (if buffer is full) cause *deadlock*
  ]
  )

  #head([File System, Secondary Storage])

  #subhead([Access Methods])
  - *Sequential Access*: Data accessed in order, beginning to end
    - *read next*: read next data, advance pointer (`read fp; fp++`)
    - *write next*: write next data, advance pointer (`write fp; fp++`)
    - *reset*: put file pointer back to the beginning (`fp=0`)
    - *skip forward/back*: fp forward/backward without reading.

  - *Direct Access*: aka Relative access. Files are fixed-length _records_. Direct access like arrays, but based on _record number_
    - *Read $n$*: return n-th data item / block
    - *Write $n$*: update n-th data item / block

  - *Indexed Access*: Use a separate _index file_ (aka _direct file_ or _index file_) contain pointers to data blocks of data file. Can direct access

  #subhead([Allocation Methods])\
  Methods should 1) Efficient disk util. 2) fast access time

  - *Contiguous Allocation*: Each file stored as a contiguous block
    - *Pro*: easy to implement, fast access time (sequential / direct)
    - *Con*: Disk fragmented, difficult to grow file
  - *Linked List Allocation*: File has many blocks, linked by pointer
    - *Pro*: No external fragmentation. File size can increase
    - *Con*: Large seek time, direct access hard, pointer overhead
  - *Indexed Allocation*: index file points to each file's first block
    - *Pro*: No external fragmentation. Supports direct access
    - *Cons*: Pointer (memory) overhead. Multi-level index (if file big)

  #subhead([UNIX I-Node Structure])
  #image("unix-i-node.png", width: 120%)

  #subhead([Disk Scheduling Algorithm])
  - Goal: Minimize seek time
  - Seek time = time taken to reach desired track

  *FCFS (First Come First Serve)*\
  *SSTF (Shortest Seek Time First)*\
  - Greedily choose the closest request from current head position.
  - Issue: starvation problem. Overhead calculating seek time.
  
  *C-SCAN (Circular SCAN)*\
  + Move from current to the largest track (read along the way)
  + Then immediately back to smallest track (*don't read*)
  + Read from smallest track to last requested track (read along way)
  #image("c-scan.png", width: 80%)

  *SCAN* (assume larger value indicator, otherwise vice versa)\
  - Minimize seek time using _Elevator algorithm_
    + Move head to the largest track (while reading along the way)
    + Then back to the smallest *requested* track (read along way)
  - *SCAN better than C-SCAN*
  #image("scan.png", width: 86%)

  *LOOK* (assume direction towards large value, if not then vice versa)\
  + Move from current to largest *requested* track (read along way)
  + Move back to smallest *requested* track (read along the way)
  - *LOOK better than SCAN*
  #image("look.png", width: 100%)

  *C-LOOK* (Circular LOOK)\
  + Move from current to largest *requested* track (read along way)
  + Immediately back to smallest *requested* track (don't read)
  + Then read to final requested track (read along way)
  #image("c-look.png", width: 100%)

  #subhead([Redundant Arrays of Inexpensive Disks (RAID)])
  #image("raid-0-1-2.png")
  #image("raid-3-4-5.png")
  
  #table(
    columns: (auto, 1fr, auto),
    inset: (x: 3.0pt, y: 2pt),
    stroke: (x: none, y: none),
    table.hline(stroke: 0.5pt + black),
    [*Level*], [*Description*], [*Util. ($N$=\#disk)*],
    table.hline(stroke: 0.5pt + black),
    [*RAID0*], [Stripe. No Fault tolerance], [$100% (N slash N)$],
    [*RAID1*], [Mirror (store 2 copies)], [$50% (N slash 2)$],
    [*RAID2*], [Bit-lvl stirpe + ECC], [Low, $(N-P)  N$],
    [*RAID3*], [Byte-lvl stripe. dedicate parity disk], [$(N-1) slash N$],
    [*RAID4*], [Block-lvl stripe. dedicate parity disk], [$(N-1) slash N$],
    [*RAID5*], [Block-lvl. distributed parity disk], [$(N-1) slash N$],
    table.hline(stroke: 0.5pt + black),
  )
  $P$ (ECC disks) must satisfy: $2^P >= N + P + 1$

  #head([Protection])
  
  _Principle of Least Privilege_: 
  - Programs, user, systems given *just enough* to perform their task
  - Limit damage if bug or get abused
  Separate _policy_ from _mechanism_: 
  - _Mechanism_: the actual implementation that enforces the _policy_
  - _Policy_: customizable security rules realized through _mechanism_

  *Domain* specifies a set of object and their operations
  - `Access-right = <object-name, right-set>`
  #image("domain.png", width: 100%)
  #grid(
    columns: 2, gutter: 1em,
    [
      *Access Matrix*
      - Row represent _domains_
      - Column represent _objects_
      - $"Access"(i, j)$ tells the set of operation $"Domain"_i$ can perform on $"Object"_j$
    ],
    image("access-matrix.png", width: 100%)
  )
  - Entire table: stored as _*ordered triples*_
    - Result: List of triples `<domain, object, right>`
  - Each column called _*access control list*_ 
    - Result: per-object list of ordered pairs: `<domain, right>`
  - Each row called _*capability list*_

#text(size: 0.96em)[```bash
fill() {
    local size=$1 content=$2 blob=()
    for ((i=0; i<size; i++)); do blob+=("$content"); done
    echo "${blob[@]}"
}

lookup() {
    local mode=$1 query=$2 index_only=$3 keys vals i results=()
    if [[ $mode == "name" ]]; then
        keys=("${emp_nms[@]}")
        vals=("${emp_ids[@]}")
    else
        keys=("${emp_ids[@]}")
        vals=("${emp_nms[@]}")
    fi
    for ((i=0; i<total_emps; i++)); do
        if [[ ${keys[i]} == $query ]]; then
            if (( index_only )); then results+=("$i");
            else results+=("${vals[i]}"); fi
        fi
    done
    echo "${results[@]}"
}

push_unique() {
    local i
    for i in ${queue[@]}; do
        (($i == $1)) && return
    done
    queue+=($1)
}

check() {
    [ -e "$1" ] && [ -f "$1" ] && [ -r "$1" ] || { echo "Invalid File: $1"; exit 1; }
}

parse() {
    local flag=0
    for p in "${params[@]}"; do
        [[ $p == "employee" ]] && { flag=1; continue; }
        if (( flag )); then
            emps+=("$p")
        else
            [[ " ${dats[*]} " != *" $p "* ]] && dats+=("$p")
        fi
    done
    n_emps=${#emps[@]}
}

rd() {
    # read employee
    while IFS= read -r line || [[ -n "$line" ]]; do
        read id nm <<< "$line"; emp_ids+=("$id"); emp_nms+=("$nm")
    done < "employees.dat"
    total_emps=${#emp_ids[@]} # total number of employees

    # read data files
    for file in "${dats[@]}"; do
        check "$file" # check file validity
        local firstline=1 index=-1 date=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            if (( firstline )); then
                firstline=0
                local tmp1 dpt month year; read tmp1 dpt month year <<< "$line"
                date=$(parse_date "$year" "$month")
                # find the index of the department
                local i
                for (( i=0; i<${#departments[@]}; i++ )); do
                    if [[ ${departments[i]} == $dpt ]]; then index=$i; break; fi
                done
                # if department not found, add the department name
                # then expand the attendance status array size
                # then initialize new slots with "N/A"
                if (( index == -1 )); then
                    departments+=("$dpt")
                    status+=($(fill "$total_emps" "N/A"))
                    last_modified+=($(fill "$total_emps" 0))
                    index=$((${#departments[@]} - 1))
                fi
            else
                local id attd; read id attd <<< "$line"
                local i=$(lookup "id" "$id" 1) # look up the index of ID
                local j=$(( index * total_emps + i ))
                if (( date >= ${last_modified[j]} )); then last_modified[j]=$date; status[j]=$attd; fi
            fi
        done < "$file"
    done
}

# echo results
out() {
    # remove redundancy
    for e in "${emps[@]}"; do
        # combine indices from both name and id searches in one array
        local found_indices=($(lookup "name" "$e" 1) $(lookup "id" "$e" 1))
        if (( ${#found_indices[@]} == 0 )); then
            echo "Invalid Entry: $e"
            exit 1
        fi
        for index in "${found_indices[@]}"; do
            push_unique "$index"
        done
    done

    # loop through employees' indices
    for ei in "${queue[@]}"; do
        echo "Attendance Report for ${emp_ids[ei]} ${emp_nms[ei]}"
        local total_present=0 total_absent=0 total_leave=0

        # loop through known departments
        for di in "${!departments[@]}"; do
            local j=$(( di * total_emps + ei ))
            local department_name=${departments[di]}
            local department_status=${status[j]}

            if [[ $department_status == "N/A" ]]; then
                echo "Department ${department_name}: ${department_status}"
            else
                echo "Department ${department_name}: ${department_status} 1"
            fi

            # increment counters based on status
            [[ $department_status == "Present" ]] && ((total_present++))
            [[ $department_status == "Absent" ]] && ((total_absent++))
            [[ $department_status == "Leave" ]] && ((total_leave++))
        done
        echo

        # output statistics
        echo "Total Present Days: $total_present"
        echo "Total Absent Days: $total_absent"
        echo "Total Leave Days: $total_leave"
        echo
    done
}

main() {
    parse
    rd
    out
}

```]
]