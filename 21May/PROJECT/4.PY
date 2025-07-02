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

void init_ports() {
	lcd_port_config();
	lcd_set_4bit();
	DDRC = 0xFF; // Set PORTC as output for LCD
	DDRE &= ~(1 << PE7); // Set PE7 as input for the button
	PORTE |= (1 << PE7); // Enable pull-up resistor on PE7
}

int main() {
	lcd_init();
	init_ports();

	while (1) {
		uint8_t button_state = PINE & (1 << PE7); // Read button state

		if (!button_state) {
			lcd_clear();
			lcd_string(1, 4, "PRESSED");
			} else {
			lcd_clear();
			lcd_string(1, 4, "NOT PRESSED");
		}

		_delay_ms(100); // Add a small delay to prevent flickering
	}
}