section .text
global mult
mult:
    push rbp
    mov rbp, rsp
    sub rsp, 30
    ; Moving parameters to stack
    ; - a
    mov dword [rbp - 4], edi
    ; - b
    mov dword [rbp - 8], esi
    ; Assign statement
    cmp dword [rbp - 8], dword 0
    mov byte [rbp - 10], byte 1
    je .L0
    mov byte [rbp - 10], byte 0
.L0:
    mov r10b, byte [rbp - 10]
    mov byte [rbp - 9], r10b
    ; If statement
    cmp byte [rbp - 9], byte 0
    je .L1
    ; If true
    ; Return statement
    mov eax, dword 0
    jmp .return
    ; If false
.L1:
    ; Assign statement
    mov r10d, dword [rbp - 8]
    mov dword [rbp - 18], r10d
    sub dword [rbp - 18], dword 1
    mov r10d, dword [rbp - 18]
    mov dword [rbp - 14], r10d
    ; Assign statement
    ; Call
    mov edi, dword [rbp - 4]
    mov esi, dword [rbp - 14]
    mov rax, qword 0
    call mult
    mov dword [rbp - 22], eax
    ; Assign statement
    mov r10d, dword [rbp - 4]
    mov dword [rbp - 30], r10d
    mov r10d, dword [rbp - 22]
    add dword [rbp - 30], r10d
    mov r10d, dword [rbp - 30]
    mov dword [rbp - 26], r10d
    ; Return statement
    mov eax, dword [rbp - 26]
    jmp .return
.return:
    mov rsp, rbp
    pop rbp
    ret

section .text
global get_byte
get_byte:
    push rbp
    mov rbp, rsp
    sub rsp, 0
    ; Moving parameters to stack
    ; No parameters
    ; Return statement
    mov al, byte -100
    jmp .return
.return:
    mov rsp, rbp
    pop rbp
    ret

extern printf

section .data
fmt:
    db 37
    db 100
    db 10
    db 0

section .text
global main
main:
    push rbp
    mov rbp, rsp
    sub rsp, 9
    ; Moving parameters to stack
    ; No parameters
    ; Assign statement
    ; Call
    mov rax, qword 0
    call get_byte
    mov byte [rbp - 1], al
    ; Assign statement
    movsx r10d, byte [rbp - 1]
    mov dword [rbp - 9], r10d
    mov r10d, dword [rbp - 9]
    mov dword [rbp - 5], r10d
    ; Discard statement
    ; Call
    mov rdi, fmt
    mov esi, dword [rbp - 5]
    mov rax, qword 0
    call printf
    ; Return statement
    mov eax, dword 0
    jmp .return
.return:
    mov rsp, rbp
    pop rbp
    ret

section .note.GNU-stack
