# Copyright (c) Agendaless Computing.
#
#  License
#
# A copyright notice accompanies this license document that identifies the copyright holders.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#     Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
#     Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
#     Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
#     If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.
#
# Disclaimer
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class reify(object):
    '''
    Put the result of a method which uses this (non-data) descriptor decorator
    in the instance dict after the first call, effectively replacing the
    decorator with an instance variable.

    It acts like @property, except that the function is only ever called once;
    after that, the value is cached as a regular attribute. This gives you lazy
    attribute creation on objects that are meant to be immutable.

    Taken from the `Pyramid project <https://pypi.python.org/pypi/pyramid/>`_.
    Modified for LiSE to make it work with __slots__ around October 2016.

    '''
    __slots__ = ['func', 'reified']

    def __init__(self, func):
        self.func = func
        self.reified = {}

    def __get__(self, inst, cls):
        if inst is None:
            return self
        if id(inst) in self.reified:
            return self.reified[id(inst)]
        self.reified[id(inst)] = retval = self.func(inst)
        return retval

    def __set__(self, inst, val):
        if id(inst) not in self.reified:
            # shouldn't happen, but it's easy to handle
            self.reified[id(inst)] = self.func(inst)
        self.reified[id(inst)].update(val)