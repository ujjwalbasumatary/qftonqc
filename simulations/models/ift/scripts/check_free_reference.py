"""Recompute the manuscript free-Ising calibration from the saved 320-site run.

Usage: python check_free_reference.py COMPANION_REPOSITORY OUTPUT_DIRECTORY
Requires NumPy. It reads existing run.log and CSV files without altering them.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repository', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
run = args.repository / 'results/ift/2026-09-24-integrable-320'
sources = {name: run / relative for name, relative in {
    'energy': 'data/energy_exp.csv', 'times': 'data/times.csv',
    'log': 'run.log', 'parameters': 'plots/summary.json',
}.items()}
parameters = json.loads(sources['parameters'].read_text())['parameters']
assert parameters['h_z'] == 0 and parameters['h_x'] > 1
times = np.loadtxt(sources['times'], delimiter=',')
full_energy = np.loadtxt(sources['energy'], delimiter=',')
assert full_energy.shape == (len(times), parameters['length'])
assert np.all(full_energy[:, -1] == 0)
energy = full_energy[:, :-1]
positions = np.arange(1, energy.shape[1] + 1, dtype=float) + 0.5
regions = (positions < parameters['length']/2, positions >= parameters['length']/2)
centroids = np.column_stack([(energy[:, region] @ positions[region]) /
                            np.sum(energy[:, region], axis=1) for region in regions])
mask = (times >= 0) & (times <= 20)
assert np.sum(mask) == 201
fits = []
for column, region in enumerate(regions):
    velocity, intercept = np.polyfit(times[mask], centroids[mask, column], 1)
    fit_residual = centroids[mask, column] - (intercept + velocity*times[mask])
    fits.append(dict(first_bond=int(np.flatnonzero(region)[0]+1),
                     last_bond=int(np.flatnonzero(region)[-1]+1),
                     velocity=float(velocity), intercept=float(intercept),
                     maximum_fit_residual=float(np.max(abs(fit_residual)))))
hx, k = parameters['h_x'], parameters['kappa']
exact_energy = 2*np.sqrt(1+hx*hx-2*hx*np.cos(k))
exact_velocity = 4*hx*np.sin(k)/exact_energy
log = sources['log'].read_text()
match = re.search(r'Incoming excitation energies at k = ±0\.38: \[([^;]+); ([^;]+);;\]', log)
assert match, 'Expected saved excitation-energy line is absent'
energies = [float(match[1]), float(match[2])]
record = dict(parameters=parameters,
              sources={name: dict(path=str(path.relative_to(args.repository)),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                       for name, path in sources.items()},
              omitted_column='The verified zero column at one-based index 320',
              position='Bond n is assigned x=n+1/2',
              weighting='Signed vacuum-subtracted bond energy, without smoothing or clipping',
              fit_time_interval=[0,20], fit_samples=int(np.sum(mask)),
              fits=fits, exact_energy=float(exact_energy), excitation_energies=energies,
              absolute_energy_errors=[float(e-exact_energy) for e in energies],
              relative_energy_errors=[float((e-exact_energy)/exact_energy) for e in energies],
              exact_central_velocity=float(exact_velocity),
              relative_speed_errors=[float(abs(f['velocity'])/exact_velocity-1) for f in fits],
              interpretation='Exact central group velocity compared with an energy-weighted finite-width packet speed. This is not a measurement of scattering delay.')
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'ift_free_reference.json').write_text(json.dumps(record, indent=2)+'\n')
with (args.output/'ift_free_reference_centroids.csv').open('w', newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['time','left_energy_centroid','right_energy_centroid'])
    writer.writerows(zip(times,centroids[:,0],centroids[:,1]))
print(json.dumps({key:record[key] for key in ('fits','exact_energy','excitation_energies','relative_energy_errors','exact_central_velocity','relative_speed_errors')},indent=2))
