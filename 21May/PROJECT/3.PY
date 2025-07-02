/*
 * LCD.c
 *
 * Created: 13-02-2025 15:10:48
 * Author : shukla
 */ 
#define F_CPU 14745600
#include <avr/io.h>
#include <util/delay.h>
#include "lcd.c"


int main()
{
    lcd_port_config();
	lcd_set_4bit();
	lcd_init();
	lcd_cursor(1,5);
	for(int i=5;i<=50;i+=5)
	{
		lcd_cursor(1,11);
		lcd_string(1,5,"Table:");
		lcd_numeric_value(1,11,i,2);
		_delay_ms(1000);
	}
    while (1);
}