/*
 * LCD2.c
 *
 * Created: 24-04-2025 2:28
 * Author : shukla
 */  


#define F_CPU 14745600
#include <avr/io.h>
#include "lcd.h"
#include <util/delay.h>


void init_ports()
{
	lcd_port_config();
	lcd_set_4bit();
	DDRC = 0xFF;
	DDRE &= ~(1 << PE7);
	PORTE |= (1 << PE7); 
}

int main()
{
	lcd_init(); 
	init_ports(); 

	uint8_t state = 0;
	uint8_t prev_button_state = 1;

	while (1)
	{
		uint8_t button_state = PINE & (1 << PE7);

		if (!button_state && prev_button_state) 
		{ 
			if (!(PINE & (1 << PE7)))
			{
				state = !state;
				lcd_clear();
			}
		}
		prev_button_state = button_state;

		if (state)
		{
			lcd_string(1, 4, "PRESSED");
		}
		else
		{
			lcd_string(1, 4, "NOT PRESSED");
		}
	}
}