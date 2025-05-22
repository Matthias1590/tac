section .text
global add
add:
    push rbp
    mov rbp, rsp
    sub rsp, 12

    ; body
    mov r10, rdi
    add r10, rsi
    mov dword [rbp - 4], r10
    mov rax, dword [rbp - 4]
    jmp .return

.return:
    mov rsp, rbp
    pop rbp
    ret
section .text
global mult
mult:
    push rbp
    mov rbp, rsp
    sub rsp, 21

    ; body
    mov r10, rsi
    cmp r10, 0
    sete r10b
    mov byte [rbp - 1], r10b
    cmp byte [rbp - 1], 0
    mov r10, .1
    je r10
    mov rax, 0
    jmp .return
.1:
    mov r10, rsi
    sub r10, 1
    mov dword [rbp - 5], r10
    mov rdi, rdi
    mov rsi, dword [rbp - 5]
    call mult
    mov dword [rbp - 9], rax
    mov r10, rdi
    add r10, dword [rbp - 9]
    mov dword [rbp - 13], r10
    mov rax, dword [rbp - 13]
    jmp .return

.return:
    mov rsp, rbp
    pop rbp
    ret
section .text
global main
main:
    push rbp
    mov rbp, rsp
    sub rsp, 4

    ; body
    mov rdi, 5
    mov rsi, 10
    call mult
    mov dword [rbp - 4], rax
    mov rax, dword [rbp - 4]
    jmp .return

.return:
    mov rsp, rbp
    pop rbp
    ret
