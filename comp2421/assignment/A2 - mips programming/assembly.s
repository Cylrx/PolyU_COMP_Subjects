#########################################################################################
# Name: WANG YUQI																		#
# Student ID:																			#
#																						#
# README:																				#
#	- Conversion Loops:																	#
#		- `b_loop`: Generates the binary representation of the input number.			#
#		- `q_loop`: Generates the quaternary representation of the input number.		#
#  		- `o_loop`: Generates the octal representation of the input number.				#
#																						#
#	- Conversion Algorithm:																#
#  		- Input number masked with [binary: 1, quaternary: 2, octal: 7] via bitwise AND #
#  		- Extracted bits are stored in the `buf` array.									#
#  		- Input number right shifted by [binary: 1, quaternary: 2, octal: 3] bits		#
#  		- Repeat the above steps until input number becomes zero.						#	
#																						#
#	- Output Display:																	#
#  		- The `print` function called after each loop									#
#  		- `print` iterates over elements of `buf` in reverse order						#
#																						#
#	- Initialization:																	#
#  		- The `init` function is called before each conversion loop						#
# 		- It initializes necessary registers and set `buf` to 0							#
#																						#
#	- Program Flow:																		#
#  		- The program prompts for an input number										#
#		- The program converts input to binary, quaternary, and octal					#
#		- Before each conversion, `init` is called										#
#		- After each conversion, `print` is called										#
#  		- Repeats the above until user exits (with 0)									#
#########################################################################################


		.data
buf:	.space 1280					# Declare buffer array for storing results
ask: 	.asciiz "Enter a number: "
cont: 	.asciiz "Continue? (1=Yes/0=No) "
repl: 	.asciiz "Input number is "
bye: 	.asciiz "Bye!"
ent:	.asciiz "\n"
bin:	.asciiz "Binary: "
qat:	.asciiz "Quaternary: "
oct: 	.asciiz "Octal: "

		.globl main
		.text

main:

		li 		$v0, 4				# Load syscall code for print string
		la 		$a0, ask			# Load address of "ask"
		syscall						# Print string "ask"
		li 		$v0, 5				# Load syscall code for read integer
		syscall						# Read integer from console
		move 	$t0, $v0			# $t0 <- read integer

		li 		$v0, 4				# Load syscall code for print string
		la 		$a0, repl			# Load address of "repl"
		syscall						# Print string "repl"
		li 		$v0, 1				# Load syscall code for print integer
		move 	$a0, $t0			# Move $t0 to $a0 ready for print
		syscall						# Print $t0
		li		$v0, 4				# Load syscall code for print string
		la		$a0, ent			# Load address of "ent"
		syscall

		jal		init				# Initialize for Binary Loop

b_loop:								# Binary Loop: Create Binary Representation
		beq 	$s0, $0, b_end		# while (s0 != 0){
		andi	$s2, $s0, 1			# 		s2 = s0 & 0b0001;
		sw		$s2, buf($s1)		# 		buf[s1] = s2;
		addi	$s1, $s1, 4			# 		s1 += 1;
		srl		$s0, $s0, 1			#		s0 >>= 1;
		j		b_loop				# }

b_end:
	 	la		$t6, bin			# $t6: Stores address of "Binary" string
		li		$t7, 124			# $t7: Stores location to start printing
		jal 	print				# Print binary results stored in "buf"
		jal		init				# Initialize for quaternary loop


q_loop:								# Quaternary Loop: Create quaternary Representation
		beq		$s0, $0, q_end		# while (s0 != 0){
		andi	$s2, $s0, 3			#		s2 = s0 & 0b0011
		sw		$s2, buf($s1)		#		buf[s1] = s2;
		addi	$s1, $s1, 4			#		s1 += 1;
		srl		$s0, $s0, 2			#		s0 >>= 2;
		j 		q_loop				# }

q_end:
	 	la		$t6, qat			# $t6: Stores address of "Quaternary" string
		li		$t7, 60				# $t7: Stores location to start printing
		jal		print				# Print quaternary results stored in "buf"
		jal		init				# Initialize for octal loop

o_loop:								# Octal Loop: Create octal representation
		beq		$s0, $0, o_end		# while (s0 != 0){
		andi	$s2, $s0, 7			#		s2 = s0 & 7;
		sw		$s2, buf($s1)		#		buf[s1] = s2;
		addi 	$s1, $s1, 4			#		s1 += 1;
		srl		$s0, $s0, 3			#		s0 >>= 3
		j 		o_loop				# }

o_end:
		la		$t6, oct			# $t6: Sotres address of "Octal" string
		li		$t7, 28				# $t7: Stores location to start printing
		jal		print				# Print octal results stored in "buf"

		# Check if User wish to continue:
		li		$v0, 4				# Load syscall code to print string
		la		$a0, cont			# Load address of "cont"
		syscall						# Print "Continue?"

		li		$v0, 5				# Load syscall code to read integer
		syscall						# Reads integer (1 or 0)

		bne		$v0, $0, main		# if (input != 0) repeat
		j		exit				# else exit

print: 
	 	li		$v0, 4				# Load syscall to print string
		move	$a0, $t6			# Load address of string to a0
		syscall						# Print string ("Binary", "Quaternary", or "Octal")
		move	$s7, $t7			# $s7: tracks index of "buf" BACKWARDS
		li		$v0, 1				# Load syscall code to print integers

print_loop:
		bltz	$s7, print_end		# while (s7 >= 0){
		lw		$a0, buf($s7)		# 		a0 = buf[s7];
		syscall						#		print(a0);
		addi	$s7, $s7, -4		#		s7 -= 1
		j 		print_loop			# }
		
print_end:
	 	li		$v0, 4				# Load syscall code to print string
		la		$a0, ent			# Load address of "ent"
		syscall						# Print "\n"
		jr		$ra					# Return to main loop


init: 
		move 	$s0, $t0			# $s0: creates copy of $t0 (input integer)
		li		$s1, 0				# $s1: tracks index of "buf" in main
		li		$s7, 0				# $s7: tracks index of "buf" in init

init_loop:
		bge		$s7, 128, init_end 	# while (s7 != 32){
		sw		$0, buf($s7)		# 		buf[s7] = 0;
		addi	$s7, $s7, 4			# 		s7 += 1;
		j		init_loop			# }

init_end:
		jr		$ra					# Return to conversion loop

exit:
	 	li		$v0, 4				# Load syscall code to print string
		la		$a0, bye			# Load address of "bye"
		syscall						# Prints "Bye!"
		li		$v0, 10				# Load syscall code to end program
		syscall						# Terminates Execution
	
	
	
