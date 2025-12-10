# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utils for ms_run"""
import os
import json
import socket
import ipaddress
import mindspore.log as logger

CURRENT_IP = None

def _generate_cmd(cmd, cmd_args, output_name):
    """
    Generates a command string to execute a Python script in the background, r
    edirecting the output to a log file.

    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        command = f"python {cmd} {' '.join(cmd_args)} > {output_name} 2>&1 &"
    else:
        command = f"{cmd} {' '.join(cmd_args)} > {output_name} 2>&1 &"
    return command


def _generate_cmd_args_list(cmd, cmd_args):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        return ['python'] + [cmd] + cmd_args
    return [cmd] + cmd_args


def _generate_cmd_args_list_with_core(cmd, cmd_args, cpu_start, cpu_end):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    # Bind cpu cores to this process.
    taskset_args = ['taskset'] + ['-c'] + [str(cpu_start) + '-' + str(cpu_end)]
    final_cmd = []
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        final_cmd = taskset_args + ['python'] + [cmd] + cmd_args
    else:
        final_cmd = taskset_args + [cmd] + cmd_args
    logger.info(f"Launch process with command: {' '.join(final_cmd)}")
    return final_cmd


def _generate_url(addr, port):
    """
    Generates a url string by addr and port

    """
    url = f"http://{addr}:{port}/"
    return url


def _get_local_ip(ip_address):
    """
    Get current IP address.

    """
    global CURRENT_IP
    if CURRENT_IP is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((ip_address, 0))
            CURRENT_IP = s.getsockname()[0]
            s.close()
        except Exception as e:
            raise RuntimeError(f"Get local ip failed: {e}. Please check whether an accessible address "
                               "is input by '--master_address'.")
    return CURRENT_IP


def _is_local_ip(ip_address):
    """
    Check if the current input IP address is a local IP address.

    """
    p = os.popen("ip -j addr")
    addr_info_str = p.read()
    p.close()
    current_ip = _get_local_ip(ip_address)
    if not addr_info_str:
        return current_ip == ip_address

    addr_infos = json.loads(addr_info_str)
    for info in addr_infos:
        for addr in info["addr_info"]:
            if addr["local"] == ip_address:
                logger.info(f"IP address found on this node. Address info:{addr}. Found address:{ip_address}")
                return True
    return False


def _convert_addr_to_ip(master_addr):
    """
    Check whether the input parameter 'master_addr' is IPv4. If a hostname is inserted, it will be converted
    to IP and then set as master host's IP.

    """
    try:
        ipaddress.IPv4Address(master_addr)
        return master_addr
    except ipaddress.AddressValueError:
        try:
            ip_address = socket.gethostbyname(master_addr)
            logger.info(f"Convert input host name:{master_addr} to ip address:{ip_address}.")
            return ip_address
        except socket.gaierror as e:
            raise RuntimeError(f"DNS resolution failed: {e}. Please check whether a correct host name "
                               "is input by '--master_address'.")


def _send_scale_num(url, scale_num):
    """
    Send an HTTP request to a specified URL, informing scale_num.

    """
    return ""
