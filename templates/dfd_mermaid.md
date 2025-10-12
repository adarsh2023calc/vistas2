Here is the Mermaid.js flowchart representing the data flow of the `calculate_average` function:
```mermaid
flowchart TD
    nums -->|list of numbers| calculate_average
    calculate_average -->|total = 0| calculate_average
    calculate_average -->|iterate over list| for_loop
    for_loop -->|add number to total| calculate_average
    for_loop -->|i = i + 1| for_loop
    for_loop -->|i < len(numbers)| for_loop
    calculate_average -->|average = total / len(numbers)| calculate_average
    calculate_average -->|average| print
    print -->|The average is: average| User
```