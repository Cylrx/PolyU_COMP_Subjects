#!/bin/bash

# -------------- HELPER METHODS -------------- # 
# parse_date() - parse date from year and month to integer for direct comparison
# fill() - fill an array with a given size and content
# lookup() - lookup employee ID via name, or name via ID, or just return the index
# push_unique() - push an element to the queue if it is not already in the queue
# check() - check if a file is valid (exists, regular, readable)

parse_date() {
    local year=$1 month=$2 num_month
    case $month in
        January)   num_month=1 ;;
        February)  num_month=2 ;;
        March)     num_month=3 ;;
        April)     num_month=4 ;;
        May)       num_month=5 ;;
        June)      num_month=6 ;;
        July)      num_month=7 ;;
        August)    num_month=8 ;;
        September) num_month=9 ;;
        October)   num_month=10 ;;
        November)  num_month=11 ;;
        December)  num_month=12 ;;
        *)         num_month=0 ;;
    esac
    echo $((year * 100 + num_month))
}

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

# -------------- MAIN PROCEDURES -------------- # 
# parse() - parse the input parameters
# rd() - read the employee and department files
# out() - output the attendance report
# main() - main function to call the above procedures

# -------------- GLOBAL VARIABLES -------------- #
# params - input parameters
# dats - department filenames
# emps - queried employees (name or ID)
# emp_nms - ALL employees' names (based on employees.dat)
# emp_ids - ALL employees' IDs (based on employees.dat)
# departments - KNOWN departments (based on department files passed in from params)
# n_emps - number of queried employees
# total_emps - total number of employees (based on employees.dat)
# queue - employee indices that needs printing

# status - simulated 2D array: input departments (row) x all employees (col)
# last_modified - simulated 2D array: input departments (row) x all employees (col)
#       - last_modified records the last modified date of the attendance status
#       - this is for keeping track of the most recent attendance status


params=("$@")
dats=()
emps=()
emp_nms=()
emp_ids=()
departments=()
n_emps=0
total_emps=0


queue=() # employee indices that needs printing
status=() # simulated 2D array: input departments (row) x all employees (col)

# similar as above. But records the last modified date of the attendance status
# this is for keeping track of the most recent attendance status
last_modified=() 

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

main