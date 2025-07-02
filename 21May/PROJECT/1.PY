/*
 * GccApplication1.c
 *
 * Created: 24-04-2025 2:28
 * Author : shukla
 */ 

#define F_CPU 14745600
#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>

int main(void) 
{
	DDRJ = 0xFF;
	PORTJ = 0x00;
	while (1) 
	{
		for (uint8_t i = 0; i < 8; i++) {
			PORTJ |= (1 << i);
			_delay_ms(500);
		}
		_delay_ms(500);
		PORTJ = 0x00;
		_delay_ms(500);
	}
}