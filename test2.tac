func @mult($a:i32, $b:i32) -> i32 {
    eq %1:i8, $b:i32, 0:i32;
    if %1:i8 {
        // base case, a * 0 = 0
        ret 0:i32;
    }

    // recursive case, a * b = a + (a * (b - 1))
    // return a + mult(a, b - 1)
    sub %2:i32, $b:i32, 1:i32;
    call %3:i32, @mult, $a:i32, %2:i32;
    add %4:i32, $a:i32, %3:i32;
    ret %4:i32;
}

func @get_byte() -> i8 {
    ret -100:i8;
}

extern @printf;
data @fmt = 37:u8, 100:u8, 10:u8, 0:u8;  // "%d\n\0"
// data @fmt = 37:u8, 117:u8, 10:u8, 0:u8;  // "%u\n\0"

func @main() -> i32 {
    call %1:i8, @get_byte;
    ext %2:i32, %1:i8;
    call %3:i32, @printf, @fmt, %2:i32;
    ret 0:i32;
}
