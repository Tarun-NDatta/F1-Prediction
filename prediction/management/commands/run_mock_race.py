"""
Django management command to run mock F1 race simulation
"""

import asyncio
import json
from django.core.management.base import BaseCommand
from django.conf import settings
import logging

# Import the mock race simulator
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

from live_prediction_system import MockRaceSimulator, EventType

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run mock F1 race simulation for testing and demonstration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--event',
            type=str,
            choices=[
                'Australian Grand Prix',
                'Chinese Grand Prix', 
                'Dutch Grand Prix',
                'Italian Grand Prix',
                'Azerbaijan Grand Prix',
                'Singapore Grand Prix'
            ],
            default='Dutch Grand Prix',
            help='Event to simulate (default: Dutch Grand Prix)'
        )
        parser.add_argument(
            '--speed',
            type=str,
            choices=['5min', '10min', '15min'],
            default='10min',
            help='Simulation speed (default: 10min)'
        )
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Enable interactive mode for user events'
        )
        parser.add_argument(
            '--auto-events',
            action='store_true',
            help='Automatically trigger random user events'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(f'Starting Mock Race Simulation: {options["event"]}')
        )
        
        try:
            # Create simulator
            simulator = MockRaceSimulator(
                event_name=options['event'],
                simulation_speed=options['speed']
            )
            
            self.stdout.write(f'Track: {options["event"]}')
            self.stdout.write(f'Total Laps: {simulator.total_laps}')
            self.stdout.write(f'Simulation Speed: {options["speed"]}')
            self.stdout.write(f'Lap Interval: {simulator.lap_interval:.1f} seconds')
            
            if options['interactive']:
                self.stdout.write(
                    self.style.WARNING('Interactive mode enabled - you can trigger events during the race')
                )
            
            # Run simulation
            asyncio.run(self._run_simulation(simulator, options))
            
        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('\nSimulation stopped by user')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error running simulation: {e}')
            )
            logger.error(f"Error running simulation: {e}")
            raise

    async def _run_simulation(self, simulator, options):
        """Run the actual simulation"""
        
        # Auto-events setup
        if options['auto_events']:
            # Schedule some automatic events
            mid_race_lap = simulator.total_laps // 2
            late_race_lap = int(simulator.total_laps * 0.7)
            
            simulator.add_user_event(EventType.WEATHER_CHANGE, mid_race_lap)
            simulator.add_user_event(EventType.SAFETY_CAR, late_race_lap)
            
            self.stdout.write(
                self.style.SUCCESS(f'Auto-events scheduled: Weather change (lap {mid_race_lap}), Safety car (lap {late_race_lap})')
            )
        
        # Main simulation loop
        while True:
            lap_result = await simulator.simulate_lap()
            
            # Display lap information
            if lap_result["status"] == "running":
                self.stdout.write(
                    f'Lap {lap_result["current_lap"]}/{lap_result["total_laps"]} - '
                    f'Leader: {lap_result["race_state"]["driver_positions"][0]["driver_name"]} - '
                    f'ML Updates: {"Active" if lap_result["ml_updates_active"] else "Stopped"}'
                )
                
                # Show recent commentary
                for comment in lap_result["commentary"]:
                    if comment["lap"] == lap_result["current_lap"]:
                        self.stdout.write(
                            self.style.SUCCESS(f'  📻 {comment["message"]}')
                        )
                
                # Interactive event handling
                if options['interactive'] and lap_result["can_add_events"]:
                    await self._handle_interactive_events(simulator, lap_result["current_lap"])
                
            elif lap_result["status"] == "finished":
                self.stdout.write(
                    self.style.SUCCESS('\n🏁 RACE FINISHED! 🏁')
                )
                break
            
            # Wait for next lap
            await asyncio.sleep(simulator.lap_interval)
        
        # Display final results
        self._display_final_results(simulator)

    async def _handle_interactive_events(self, simulator, current_lap):
        """Handle interactive event input (simplified for CLI)"""
        # In a real implementation, this would be handled by the web interface
        # For CLI demo, we'll just show available options
        if current_lap % 10 == 0:  # Every 10 laps, show options
            self.stdout.write(
                f'  💡 Interactive events available: {simulator.max_user_events - simulator.user_events_used} remaining'
            )

    def _display_final_results(self, simulator):
        """Display final race results"""
        results = simulator.get_final_results()
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(f'FINAL RESULTS - {simulator.event_name}')
        self.stdout.write('='*60)
        
        for result in results:
            status = " (DNF)" if result["dnf"] else ""
            penalty = f" (+{result['penalties']}s)" if result["penalties"] > 0 else ""
            
            self.stdout.write(
                f'P{result["position"]:2d}: {result["driver"]:20s} '
                f'({result["pit_stops"]} stops){penalty}{status}'
            )
        
        # Summary statistics
        summary = simulator.get_simulation_summary()
        self.stdout.write(f'\nTotal Commentary Messages: {len(summary["commentary_feed"])}')
        self.stdout.write(f'User Events Used: {summary["user_events_used"]}/{simulator.max_user_events}')
        self.stdout.write(f'Track Type: {summary["track_config"]["track_type"].value}')
        
        self.stdout.write('\n' + '='*60)
