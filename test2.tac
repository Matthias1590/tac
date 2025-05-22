func @mult($a:i32, $b:i32) -> i32 {
    %1:i8 = eq $b:i32, 0:i32;
    if %1:i8 {
        // base case, a * 0 = 0
        ret 0:i32;
    }

    // recursive case, a * b = a + (a * (b - 1))
    // return a + mult(a, b - 1)
    %2:i32 = sub $b:i32, 1:i32;
    %3:i32 = call i32, @mult, $a:i32, %2:i32;
    %4:i32 = add $a:i32, %3:i32;
    ret %4:i32;
}

func @get_byte() -> i8 {
    ret -100:i8;
}

extern @printf;
data @fmt = 37:u8, 100:u8, 10:u8, 0:u8;  // "%d\n\0"
// data @fmt = 37:u8, 117:u8, 10:u8, 0:u8;  // "%u\n\0"

func @main() -> i32 {
    // %1:i32 = call i32, @mult, 5:i32, 3:i32;
    // call i32, @printf, @fmt, %1:i32;
    %1:i8 = call i8, @get_byte;
    %2:i32 = ext i32, %1:i8;
    call i32, @printf, @fmt, %2:i32;
    ret 0:i32;
}
