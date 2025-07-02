/*
 * Experiment2.c
 *
 * Created: 13-02-2025 14:02:08
 * Author : shail
 */ 

#define F_CPU 14745600  //for delay calculation F_CPU defines the clock frequency of the microcontroller (14.7456 MHz).This value is required for timing-related functions like _delay_ms().

#include <avr/io.h>         //avr/io.h: This library is used to access input/output (I/O) registers of the microcontroller.
#include <util/delay.h>     //util/delay.h: Provides delay functions like _delay_ms() to create time delays.


void buzzer_on(void) {
	PORTC |= 0x08;  //setting port c high
}

void buzzer_off(void) {
	PORTC &= 0xF7;   //for setting port c low
}

void port_init(void) {
	DDRC |= 0x08;    //setting PC3 as o/p buzzer
	PORTC &= 0xF7;	 // buzzer off initially
	DDRE &= 0x7F;    //setting PE7 as i/p(boot switch)
	PORTE |= 0x80;   //enabling pull up register
}

int main(void) {
	port_init();

	while (1) {
		if (!(PINE & 0x80)) {
			buzzer_on();
			_delay_ms(200);
			buzzer_off();
			_delay_ms(200);
			} else {
			buzzer_off();
		}
	}
}
