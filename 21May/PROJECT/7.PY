/*
 * LAB_7.c
 *
 * Created: 24-04-2025 2:28
 * Author : shukla
 */ 
#define F_CPU 14745600
#include <avr/io.h>
#include <util/delay.h>
#include "lcd.h"

void sensor_init()
{
	ADCSRA = 0x00;
	ADCSRB = 0x00;
	ADMUX = 0x20;
	ACSR = 0x80;
	ADCSRA = 0x86;
}

unsigned char read_sensor(unsigned char channel) {
	unsigned char result;
	if (channel > 7) {
		ADCSRB = 0x08;
	}
	channel = channel & 0x07;
	ADMUX = 0x20 | channel;
	ADCSRA |= 0x40;
	while ((ADCSRA & 0x10) == 0);
	result = ADCH;
	ADCSRA |= 0x10;
	ADCSRB = 0x00;
	return result;
}

unsigned int calculate_distance(unsigned char raw_value) {
	float distance;
	if (raw_value < 10) return 800;
	distance = 10.00 * (2799.6 * (1.00 / pow((double)raw_value, 1.1546)));
	if (distance > 800) distance = 800;
	if (distance < 30) distance = 30;
	return (unsigned int) distance;
}

void display_obstacle_status(unsigned int distance) {
	lcd_cursor(2, 1);
	if (distance >= 100 && distance < 200) {
		lcd_string(2, 1, "Obstacle is Near ");
		} else if (distance >= 200 && distance < 500) {
		lcd_string(2, 1, "Obstacle is Far  ");
		} else if (distance >= 500 && distance < 800) {
		lcd_string(2, 1, "Obstacle is Very Far");
		} else {
		lcd_string(2, 1, "No Obstacle      ");
	}
}

int main() {
	lcd_port_config();
	lcd_set_4bit();
	lcd_init();
	sensor_init();
	while (1) {
		unsigned char sensor_value = read_sensor(11);
		unsigned int measured_distance = calculate_distance(sensor_value);
		lcd_cursor(1, 1);
		lcd_string(1, 1, "ADC:");
		lcd_numeric_value(1, 6, sensor_value, 4);
		lcd_cursor(2, 1);
		lcd_string(2, 1, "Dist:");
		display_obstacle_status(measured_distance);
		_delay_ms(200);
	}
}