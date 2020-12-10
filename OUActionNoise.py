'''
file:       OUActionNoise.py
context:    TORCS project, team 1 in CSC450, taught by Dr. Razib Iqbal. 
            Add noise for AI bot exploration
            
project collaborators:  Blake Engelbrecht
                        David Engleman
                        Shannon Groth
                        Khaled Hossain
                        Jacob Rader

LICENSE INFORMATION:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

class OUActionNoise(object):

    def generate_noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

