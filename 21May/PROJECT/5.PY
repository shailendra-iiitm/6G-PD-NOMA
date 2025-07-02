/*
 * GccApplication3.c
 *
 * Created: 24-04-2025 2:28
 * Author : shukla
 */ 
#define F_CPU 14745600
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

void motion_pin_config(void)
{
	DDRA = DDRA | 0x0F;
	PORTA = PORTA & 0xF0;
	DDRL = DDRL | 0x18;
	PORTL = PORTL | 0x18;
}

void forward(void)
{
	PORTA = 0x00;
	PORTA = 0x06;
}

void left(void)
{
	PORTA = 0x00;
	PORTA = 0x05;
}

void right(void)
{
	PORTA = 0x00;
	PORTA = 0x0A;
}

void stop(void)
{
	PORTA = 0x00;
	PORTA = 0x00;
}

void trace_S_path(void)
{
	forward();
	_delay_ms(2500);

	left();
	_delay_ms(900);

	forward();
	_delay_ms(2500);
	stop();
	left();
	_delay_ms(950);

	forward();
	_delay_ms(2500);
	
	right();
	_delay_ms(950);

	forward();
	_delay_ms(2500);

	right();
	_delay_ms(950);

	forward();
	_delay_ms(2500);

	stop();
}

int main(void)
{
	motion_pin_config();
	
	_delay_ms(3000);
	
	trace_S_path();
	
	return 0;
}