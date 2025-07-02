/*
 * EXP 6.c
 *
 * Created: 24-04-2025 2:28
 * Author : shukla
 */ 

#define F_CPU 14745600
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include "lcd.h"

void motion_pin_config(void)
{
	DDRA  = 0x0F;
	PORTA = 0x00;
	DDRL  = 0x18;
	PORTL = 0x18;
}
void boot_switch_config(void)
{
	DDRE &= ~(1 << PE7);
	PORTE |= (1 << PE7);
}
void port_init(void)
{
	motion_pin_config();
	boot_switch_config();
}
void motion_timer5_init()
{
	TCCR5B = 0x00;
	TCNT5H = 0xFF;
	TCNT5L = 0x01;
	OCR5AL = 0xFF;
	OCR5BL = 0xFF;
	TCCR5A = 0xA9;
	TCCR5B = 0x0B;
}
void velocity (unsigned char left_motor, unsigned char right_motor)
{
	OCR5AL = (unsigned char)left_motor;
	OCR5BL = (unsigned char)right_motor;
}
void forward(void)
{
	PORTA = 0x00;
	PORTA = 0x06;
}
void stop(void)
{
	PORTA = 0x00;
	PORTA = 0x00;
}
void motion_control(void)
{
	int speed = 0;
	int direction = 0;
	lcd_string(1,3,"Speed:");
	lcd_numeric_value(1,10,speed,3);
	stop();
	while(1)
	{
		if(PINE & (1 << PE7))
		{
			_delay_ms(50);
			while(PINE & (1 << PE7));
			_delay_ms(50);
			if(speed==0)
			{
				direction=0;
			}
			else if (speed == 252)
			{
				direction=1;
			}
			if(direction == 0)
			{
				speed += 63;
			}
			else
			{
				speed -= 63;
			}
			lcd_numeric_value(1,10,speed,3);
			
			if(speed>0)
			{
				forward();
				velocity(speed,speed);
			}
			else
			{
				stop();
			}
		}
	}
}
int main()
{
	lcd_port_config();
	lcd_init();
	motion_pin_config();
	boot_switch_config();
	motion_timer5_init();
	motion_control();
	return 0;
}